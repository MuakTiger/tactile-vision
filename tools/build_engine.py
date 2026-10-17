#!/usr/bin/env python3
"""
Depth Anything V2 metric 모델 → ONNX → TensorRT FP16 엔진.

기존 파이프라인이 518×518 정사각을 쓰던 문제를 여기서 바로잡는다.
  - 1280×720을 518×518로 강제 리사이즈하면 가로가 44% 압축된다.
  - metric 추정은 암묵적으로 초점거리를 가정하므로 기하 왜곡이 거리 오차로 직결된다.
  - 따라서 카메라 크롭 종횡비와 엔진 입력 종횡비를 일치시킨다.

입력 크기는 세 제약을 동시에 만족해야 한다 (geometry.InputSpec이 검증):
  (a) DINOv2 패치 14의 배수
  (b) 폭이 그리드 열 수(6)의 배수
  (c) 높이가 그리드 행 수(4)의 배수

사용:
  python tools/build_engine.py --spec wide
  python tools/build_engine.py --spec faithful --outdoor
  python tools/build_engine.py --spec wide_hi --relative   # 상대깊이(실외 땅 기준 환산용)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision import geometry as geo  # noqa: E402

# 이 프로젝트는 외부 디렉토리에 어떤 파일도 쓰지 않는다.
# 모델 정의는 vendor/ 에 복사해 두었고, 체크포인트도 이 프로젝트 안에 둔다.
# (외부 트리를 import하면 그쪽에 __pycache__가 생성된다)
VENDOR_ROOT = REPO / "vendor"
CKPT_DIR = REPO / "checkpoints"

CKPTS = {
    "indoor": ("depth_anything_v2_metric_hypersim_vits.pth", 20.0),
    "outdoor": ("depth_anything_v2_metric_vkitti_vits.pth", 80.0),
    # 상대깊이(metric 미세조정 전). 출력은 단위 없는 상대 역깊이 → ground.py가 땅 기준으로 m 환산
    "relative": ("depth_anything_v2_vits.pth", None),
}
HF_REPO = {
    "indoor": "Depth-Anything-V2-Metric-Hypersim-Small",
    "outdoor": "Depth-Anything-V2-Metric-VKITTI-Small",
    "relative": "Depth-Anything-V2-Small",
}
VITS_CFG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_engine")


def export_onnx(spec: geo.InputSpec, domain: str, onnx_path: Path) -> float | None:
    """모델을 지정 해상도 고정 shape ONNX로 내보낸다. max_depth 반환(상대깊이는 None)."""
    import torch

    # vendor/ 의 metric 전용 dpt.py를 쓴다. 상대깊이용과 forward가 다르다:
    # metric은 sigmoid 출력에 max_depth를 곱해 미터를 직접 낸다.
    sys.path.insert(0, str(VENDOR_ROOT))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

    ckpt_name, max_depth = CKPTS[domain]
    ckpt = CKPT_DIR / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(
            f"체크포인트 없음: {ckpt}\n"
            f"다운로드:\n"
            f"  mkdir -p {CKPT_DIR}\n"
            f"  curl -L -o {ckpt} \\\n"
            f"    https://huggingface.co/depth-anything/{HF_REPO[domain]}"
            f"/resolve/main/{ckpt_name}"
        )

    log.info("모델 로드: %s (%s)", ckpt.name, "상대깊이" if max_depth is None else f"max_depth={max_depth:.0f}m")
    model = DepthAnythingV2(**{**VITS_CFG, "max_depth": max_depth})
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, spec.height, spec.width)

    # 출력이 실제로 미터 스케일인지 내보내기 전에 확인
    with torch.no_grad():
        out = model(dummy)
    log.info(
        "출력 shape=%s  범위 %.2f~%.2f %s",
        tuple(out.shape), float(out.min()), float(out.max()),
        "(상대 역깊이, 단위 없음)" if max_depth is None else f"m (max_depth {max_depth:.0f})",
    )
    if out.shape[-2:] != (spec.height, spec.width):
        raise RuntimeError(f"출력 해상도 불일치: {out.shape} vs {spec}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("ONNX 내보내기 → %s", onnx_path)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["depth_m" if max_depth is not None else "rel_disp"],
        dynamo=False,
    )
    log.info("ONNX 완료 (%.1f MB)", onnx_path.stat().st_size / 1e6)
    return max_depth


def build_trt(onnx_path: Path, engine_path: Path, fp16: bool = True) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    log.info("ONNX 파싱 중...")
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            log.error("파싱 오류 %d: %s", i, parser.get_error(i))
        raise RuntimeError("ONNX 파싱 실패")

    cfg = builder.create_builder_config()
    # Orin Nano 8GB — 워크스페이스를 과하게 잡으면 OOM. 1GB로 제한.
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
        log.info("FP16 활성화")
    else:
        log.warning("FP16 미사용 — 속도가 절반 이하로 떨어짐")

    inp = network.get_input(0)
    out = network.get_output(0)
    log.info("입력 %s %s / 출력 %s %s", inp.name, inp.shape, out.name, out.shape)

    log.info("엔진 빌드 시작 (수 분 소요)...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError("엔진 빌드 실패")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    log.info(
        "엔진 완료: %s (%.1f MB, %.0f초)",
        engine_path, engine_path.stat().st_size / 1e6, time.perf_counter() - t0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="wide", choices=sorted(geo.INPUT_SPECS),
                    help="입력 해상도 프리셋")
    ap.add_argument("--outdoor", action="store_true",
                    help="VKITTI(실외, max_depth 80m) 사용. 기본은 Hypersim(실내 20m)")
    ap.add_argument("--relative", action="store_true",
                    help="상대깊이 모델(Depth-Anything-V2-Small). 실외 땅 기준 환산(depth.outdoor_kind: ground)용")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--onnx-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="기존 파일 무시하고 재생성")
    args = ap.parse_args()

    spec = geo.INPUT_SPECS[args.spec]
    if args.relative and args.outdoor:
        ap.error("--relative와 --outdoor는 함께 쓸 수 없음")
    if args.relative:
        domain, stem = "relative", f"da2_rel_vits_{spec.width}x{spec.height}"
    else:
        domain = "outdoor" if args.outdoor else "indoor"
        tag = "vkitti" if args.outdoor else "hypersim"
        stem = f"da2_metric_{tag}_vits_{spec.width}x{spec.height}"

    onnx_path = REPO / "models" / f"{stem}.onnx"
    engine_path = REPO / "models" / f"{stem}.engine"

    print(f"\n{spec.describe()}")
    print(f"도메인: {domain} / 출력: {engine_path.name}\n")

    if not onnx_path.exists() or args.force:
        export_onnx(spec, domain, onnx_path)
    else:
        log.info("ONNX 재사용: %s", onnx_path)

    if args.onnx_only:
        return 0

    if engine_path.exists() and not args.force:
        log.info("엔진이 이미 존재: %s (--force로 재빌드)", engine_path)
        return 0

    build_trt(onnx_path, engine_path, fp16=not args.no_fp16)
    key = "outdoor_engine (+ outdoor_kind: ground)" if args.relative else "engine"
    print(f"\nconfig/default.yaml 의 depth.{key} 을 다음으로 설정:")
    print(f"  models/{engine_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
