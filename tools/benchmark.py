#!/usr/bin/env python3
"""
성능 벤치마크 — 논문의 지연 예산 표를 이 출력으로 채운다.

측정 항목
  1. 단계별 소요시간 분해 (전처리 / 추론 / 풀링 / D2H)
  2. 입력 해상도별 비교
  3. 그리드 리듀서별 비교
  4. 종단 지연 예산 (카메라 노출 → 핀 도착)
  5. 전력 모드별 비교 (선택)

사용:
  python tools/benchmark.py
  python tools/benchmark.py --iters 100 --source /tmp/synth.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision import protocol as P  # noqa: E402
from tactile_vision.config import load  # noqa: E402
from tactile_vision.depth import DepthEngine  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def stat(v: list[float]) -> str:
    a = np.array(v)
    return (
        f"평균 {a.mean():6.2f}  p50 {np.percentile(a,50):6.2f}  "
        f"p95 {np.percentile(a,95):6.2f}  최대 {a.max():6.2f}"
    )


def bench_stages(de: DepthEngine, img: np.ndarray, iters: int) -> dict[str, list[float]]:
    """단계를 분리해 각각 측정. GPU 동기화를 넣어야 정확한 값이 나온다."""
    out: dict[str, list[float]] = {"preprocess": [], "infer": [], "pool": [], "d2h": []}
    spec = de.spec

    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.stream(de.stream):
            de._preprocess(img)
        de.stream.synchronize()
        t1 = time.perf_counter()

        with torch.inference_mode(), torch.cuda.stream(de.stream):
            de.context.execute_async_v3(stream_handle=de.stream.cuda_stream)
        de.stream.synchronize()
        t2 = time.perf_counter()

        with torch.inference_mode(), torch.cuda.stream(de.stream):
            depth = de.d_output.view(spec.height, spec.width)
            cells = de._pool_cells(depth)
        de.stream.synchronize()
        t3 = time.perf_counter()

        _ = cells.cpu().numpy()
        t4 = time.perf_counter()

        out["preprocess"].append((t1 - t0) * 1000)
        out["infer"].append((t2 - t1) * 1000)
        out["pool"].append((t3 - t2) * 1000)
        out["d2h"].append((t4 - t3) * 1000)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="실제 영상으로 측정")
    args = ap.parse_args()

    cfg = load(args.config)
    spec = cfg.depth.spec

    print("=" * 74)
    print("  시각-촉각 감각대체 시스템 성능 벤치마크")
    print("=" * 74)
    print(f"  보드     Jetson Orin Nano 8GB")
    print(f"  입력     {spec.describe()}")
    print(f"  엔진     {Path(cfg.depth.engine).name}")
    print(f"  반복     {args.iters}회")
    print()

    de = DepthEngine(cfg.depth, cfg.grid, cfg.path(cfg.depth.engine))

    # 실제 영상이 있으면 그걸 쓰고, 없으면 노이즈 이미지
    if args.source:
        import cv2

        cap = cv2.VideoCapture(args.source)
        ok, raw = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"영상 읽기 실패: {args.source}")
        from tactile_vision.camera import crop_rect

        x, y, w, h = crop_rect(raw.shape[1], raw.shape[0], spec.aspect)
        img = cv2.resize(raw[y : y + h, x : x + w], (spec.width, spec.height))
    else:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (spec.height, spec.width, 3), dtype=np.uint8)

    de.warmup(15)

    print("─" * 74)
    print("1. 단계별 소요시간 (ms)")
    print("─" * 74)
    res = bench_stages(de, img, args.iters)
    total = 0.0
    for k, label in [
        ("preprocess", "전처리(GPU: BGR→RGB, 정규화)"),
        ("infer", "TRT 추론 (DA-V2 metric ViT-S FP16)"),
        ("pool", "셀 풀링 (GPU kthvalue 24셀)"),
        ("d2h", "D2H 전송 (24 float = 96바이트)"),
    ]:
        m = float(np.mean(res[k]))
        total += m
        print(f"  {label:<38} {stat(res[k])}")
    print(f"  {'합계':<38} 평균 {total:6.2f} ms  →  {1000/total:.1f} fps")

    print()
    print("─" * 74)
    print("2. 셀 풀링 위치: GPU vs CPU (전체 깊이맵 D2H + numpy 평균)")
    print("─" * 74)
    # 기존 방식: 전체 깊이맵을 CPU로 내려 numpy로 평균
    t = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dm = de.d_output.view(spec.height, spec.width).cpu().numpy()
            R, C = geo.GRID_ROWS, geo.GRID_COLS
            vals = []
            for rr in np.array_split(dm, R, axis=0):
                for cc in np.array_split(rr, C, axis=1):
                    vals.append(float(np.mean(cc)))
        t.append((time.perf_counter() - t0) * 1000)
    cpu_ms = float(np.mean(t))
    gpu_ms = float(np.mean(res["pool"])) + float(np.mean(res["d2h"]))
    bytes_full = spec.pixels * 4
    print(f"  전체맵 D2H + CPU 평균     평균 {cpu_ms:6.2f} ms  "
          f"({bytes_full/1024:.0f} KB 전송)")
    print(f"  GPU 풀링 + 24값 D2H       평균 {gpu_ms:6.2f} ms  (96 B 전송)")
    print(f"  시간 차이                 {cpu_ms - gpu_ms:+6.2f} ms/프레임"
          f"  ({cpu_ms/max(gpu_ms,1e-6):.2f}배)")
    print(f"  전송량                    {bytes_full/96:.0f}배 감소")
    print()
    print("  해석: Orin은 CPU/GPU가 DRAM을 공유하므로 551KB D2H가 이미 저렴하다.")
    print("        따라서 GPU 풀링의 이득은 '시간 절약'이 아니다.")
    print("        실제 이득은 (1) 분위수 리듀서를 값싸게 쓸 수 있다는 점,")
    print("        (2) 전송량이 5880배 적어 채널 확장·외장 GPU 이전에 유리하다는 점이다.")
    print("        dGPU(PCIe)에서는 시간 이득이 유의미해진다.")

    print()
    print("─" * 74)
    print("3. 리듀서별 비교 (같은 프레임, 셀 대표값 산출 방식)")
    print("─" * 74)
    orig = cfg.grid.reducer
    cells_by = {}
    for red in ("mean", "percentile", "min"):
        cfg.grid.reducer = red
        de.gcfg = cfg.grid
        de._k = de._percentile_k()
        t = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.cuda.stream(de.stream):
                c = de._pool_cells(de.d_output.view(spec.height, spec.width))
            de.stream.synchronize()
            t.append((time.perf_counter() - t0) * 1000)
        cells_by[red] = de.infer(img)
        print(f"  {red:<12} {stat(t)}")
    cfg.grid.reducer = orig
    de.gcfg = cfg.grid
    de._k = de._percentile_k()

    print()
    print("  같은 장면의 셀 거리 차이 (기둥/의자다리 같은 얇은 장애물 검출력)")
    mn, pc, mx = cells_by["mean"], cells_by["percentile"], cells_by["min"]
    print(f"    mean       최근접 {mn.min():.2f}m  평균 {mn.mean():.2f}m")
    print(f"    percentile 최근접 {pc.min():.2f}m  평균 {pc.mean():.2f}m")
    print(f"    min        최근접 {mx.min():.2f}m  평균 {mx.mean():.2f}m")
    print(f"    → percentile이 mean보다 평균 {float(np.mean(mn-pc)):.2f}m 가깝게 보고"
          f" (얇은 장애물을 놓치지 않음)")

    print()
    print("─" * 74)
    print("4. 위험도 검출(YOLO) 단계")
    print("─" * 74)
    from tactile_vision.hazard import HazardEncoder  # noqa: E402

    hz = HazardEncoder(cfg.hazard, spec.width, spec.height)
    if hz.model is None:
        haz_ms = 0.0
        print("  YOLO 엔진 없음 — 위험도 검출 비활성 (거리 정보만 전달)")
        print("  빌드: ultralytics로 yolov8n.engine을 models/에 생성")
    else:
        hz.warmup(5)  # 예열 없이 재면 엔진 로딩이 섞여 측정이 무의미하다
        cd = np.full(geo.CELL_COUNT, 2.0, dtype=np.float32)
        t = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            hz.encode(img, cd)
            t.append((time.perf_counter() - t0) * 1000)
        haz_ms = float(np.mean(t))
        print(f"  encode() 전체            {stat(t)}")
        print(f"  → 깊이추론과 합산 {total + haz_ms:.2f} ms "
              f"({1000/(total+haz_ms):.1f} fps)")
        print()
        print("  주의: ultralytics는 첫 predict 호출에서 predictor 구성 +")
        print("        TRT 엔진 역직렬화를 수행한다(수 초). 예열하지 않으면")
        print("        초기 프레임이 실시간 경로를 막고 통계가 왜곡된다")
        print("        (실측: 60프레임 평균이 37ms → 180ms로 튐).")

    print()
    print("─" * 74)
    print("5. 종단 지연 예산 (카메라 노출 → 핀 도착)")
    print("─" * 74)
    cam_ms = 1000.0 / cfg.camera.fps
    tx_ms = P.GRID_FRAME_BYTES * 10 / cfg.transport.baud * 1000
    prof = cfg.mapping.profile
    servo_1lv = geo.travel_time_s(prof.step_mm) * 1000
    servo_full = geo.travel_time_s(cfg.mapping.stroke_mm) * 1000
    fw_ms = 5.0  # 펌웨어 서보 갱신 주기
    pin_ms = 1000.0 / cfg.transport.pin_rate_hz  # Pico v3 핀 갱신 박자
    rows = [
        ("카메라 노출 + USB 전송", cam_ms, f"{cfg.camera.fps}fps 프레임 주기 상한"),
        ("깊이 파이프라인", total, "실측 (전처리+추론+풀링+D2H)"),
        ("위험도 검출", haz_ms, "YOLO, 비활성 시 0"),
        ("시리얼 전송", tx_ms, f"{P.GRID_FRAME_BYTES}B, USB CDC라 실제는 더 짧음"),
        ("Pico 박자 대기 (평균)", pin_ms / 2,
         f"박자 {pin_ms:.0f}ms의 절반, 최악 {pin_ms:.0f}ms (산정치)"),
        ("펌웨어 반영 주기", fw_ms, "200Hz 서보 갱신"),
        ("서보 1단계 이동", servo_1lv, f"{prof.step_mm:.2f}mm @ {geo.SG90_SPEED_MM_PER_S:.0f}mm/s"),
    ]
    acc = 0.0
    for name, ms, note in rows:
        acc += ms
        print(f"  {name:<26} {ms:7.1f} ms   ({note})")
    print(f"  {'─'*26} {'─'*7}")
    print(f"  {'1단계 변화 종단 지연':<26} {acc:7.1f} ms   (최악 +{pin_ms / 2:.0f}ms)")
    print(f"  {'전체 스트로크 이동 시':<26} {acc - servo_1lv + servo_full:7.1f} ms"
          f"   (서보 {servo_full:.0f}ms)")
    compute = total + haz_ms
    print()
    print(f"  병목: 서보 기계적 이동 ({servo_full:.0f}ms) > 연산 전체 ({compute:.1f}ms)"
          f" — {servo_full/max(compute,1e-6):.1f}배")
    print(f"  → 연산을 {1000/compute:.0f}fps로 돌려도 촉각 갱신은 서보와 Pico 박자"
          f"({cfg.transport.pin_rate_hz:g}Hz)가 정한다.")
    print(f"  → 박자 {pin_ms:.0f}ms 동안 새 결과 약 {pin_ms / compute:.1f}개 "
          f"(연산 {compute:.1f}ms, 장면 판별·카메라 대기 제외) — Pico는 그중 최신 하나만 적용")
    print(f"  → 박자 {pin_ms:.0f}ms 안에 서보 {pin_ms / servo_1lv:.1f}단계 이동 가능 "
          f"(1단계 {servo_1lv:.0f}ms)")

    print()
    print("─" * 74)
    print("6. 메모리")
    print("─" * 74)
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    print(f"  torch 할당 {alloc:.1f} MB / 예약 {reserved:.1f} MB")
    try:
        import subprocess

        out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
        line = [l for l in out.splitlines() if l.startswith("Mem:")][0].split()
        print(f"  시스템 RAM 총 {line[1]}MB / 사용 {line[2]}MB / 가용 {line[6]}MB")
    except Exception:  # noqa: BLE001
        pass

    print()
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
