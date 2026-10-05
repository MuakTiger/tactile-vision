#!/usr/bin/env python3
"""
위험도 검출용 YOLO TensorRT 엔진 빌드.

중요 — 출력 격리
──────────────
ultralytics의 `export()`는 **입력 .pt 파일이 있는 디렉토리에 결과를 쓴다.**
따라서 남의 작업 폴더에 있는 .pt를 직접 넘기면 그 폴더에 .onnx/.engine이
생성되고, 같은 이름의 기존 파일을 조용히 덮어쓴다.

이 스크립트는 .pt를 격리된 임시 디렉토리로 복사한 뒤 거기서 내보내고,
완성된 엔진만 이 프로젝트의 models/ 로 옮긴다. 원본 디렉토리는 읽기만 한다.

사용:
  python tools/build_yolo.py --weights /path/to/yolov8n.pt
  python tools/build_yolo.py --weights yolov8n.pt --imgsz 512 288
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_yolo")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", required=True, help=".pt 가중치 경로 (읽기만 함)")
    ap.add_argument(
        "--imgsz", nargs="+", type=int, default=[640],
        help="입력 크기. 단일값(정사각) 또는 'W H'. 기본 640",
    )
    ap.add_argument("--no-half", action="store_true", help="FP16 비활성")
    ap.add_argument("--out", default=None, help="출력 파일명 (기본: <이름>.engine)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.weights).expanduser().resolve()
    if not src.exists():
        log.error("가중치 없음: %s", src)
        return 1

    out_name = args.out or f"{src.stem}.engine"
    dst = REPO / "models" / out_name
    if dst.exists() and not args.force:
        log.info("엔진이 이미 존재: %s (--force로 재빌드)", dst)
        return 0

    # ultralytics의 imgsz는 (h, w) 순서. CLI는 직관을 위해 W H로 받는다.
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
        shape_desc = f"{imgsz}x{imgsz}"
    elif len(args.imgsz) == 2:
        w, h = args.imgsz
        imgsz = (h, w)
        shape_desc = f"{w}x{h}"
    else:
        log.error("--imgsz는 1개 또는 2개 값만 허용")
        return 1

    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="tv_yolo_") as tmp:
        tmpdir = Path(tmp)
        # 격리: .pt를 임시 폴더로 복사해 원본 폴더에 아무것도 쓰지 않는다
        local_pt = tmpdir / src.name
        shutil.copy2(src, local_pt)
        log.info("격리 빌드 디렉토리: %s", tmpdir)
        log.info("입력 %s (원본은 읽기만: %s)", shape_desc, src)

        model = YOLO(str(local_pt))
        t0 = time.perf_counter()
        produced = model.export(
            format="engine",
            half=not args.no_half,
            imgsz=imgsz,
            device=0,
            verbose=False,
        )
        log.info("빌드 %.0f초", time.perf_counter() - t0)

        produced = Path(produced)
        if not produced.exists():
            log.error("엔진 생성 실패")
            return 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, dst)

    log.info("엔진 저장: %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
    print(f"\nconfig/default.yaml 의 hazard.engine 을 다음으로 설정:")
    print(f"  models/{dst.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
