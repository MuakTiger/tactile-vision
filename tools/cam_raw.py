#!/usr/bin/env python3
"""
카메라 원본 미리보기 (처리 없음).

카메라 프레임을 크롭·리사이즈·반전·덧그림 없이 1:1 픽셀 그대로 보여준다.
화각 측정(벽에 테이프), 초점 맞추기, 장착 각도 확인용.

키
  q / ESC  종료          s  원본 프레임 저장 (logs/cam_raw/, PNG 무손실)
  a        자동초점 켜기/끄기 (자동초점 카메라만)
  [ / ]    초점값 −/+ 한 단계 (자동초점이 꺼져 있을 때)

초점
  자동초점 카메라(예: Logitech C920)는 기본으로 자동초점을 끄고 초점값을 고정한다.
  자동초점이 렌즈를 움직이면 초점거리(화각)도 조금씩 바뀌어 거리 추정과 캘리브레이션이 흔들린다.
  C920 초점값은 0~250(5 단위), 0 = 무한대 쪽, 클수록 가까운 쪽
  (2026-10-02 선명도 비교: 방 안 장면에서 0이 자동초점 결과와 같고 60은 흐려짐).
  설정은 카메라를 뽑을 때까지 유지된다. 현재 상태는 창 제목에 표시한다(영상에는 그리지 않는다).
  고정초점 카메라(WSD-ZX001)는 초점 컨트롤이 없으므로 아무것도 바꾸지 않는다.

사용
  python tools/cam_raw.py                      /dev/video0, 1920×1080 MJPG 30 fps, 자동초점 끔·초점 0
  python tools/cam_raw.py --focus 20           초점값 20으로 고정
  python tools/cam_raw.py --autofocus on       자동초점을 켠 채로
  python tools/cam_raw.py --size 1280x720      장치 파이프라인과 같은 해상도
  python tools/cam_raw.py --device 1           /dev/video1

참고: WSD-ZX001(2026-10-01)과 C920(2026-10-02) 모두 1280×720이 1920×1080 전체 화각을
축소한 것이다(두 해상도 특징점 정합: 배율 1.500 / 1.499, 이동 0). 어느 해상도든 화각은 같다.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tactile_vision.v4l2ctl import Focus, V4L2Ctrl  # noqa: E402  장치 파이프라인과 같은 초점 제어

WIN = "Camera raw"
MAX_MISSES = 300  # 연속 실패 프레임 수 (약 3 s) 넘으면 종료

log = logging.getLogger("cam_raw")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0, help="카메라 번호 (기본 0 = /dev/video0)")
    ap.add_argument("--size", default="1920x1080", help="요청 해상도 WxH (기본 1920x1080)")
    ap.add_argument("--fps", type=float, default=30.0, help="요청 fps (기본 30)")
    ap.add_argument("--autofocus", choices=("off", "on"), default="off",
                    help="자동초점 (기본 off: 끄고 --focus 값으로 고정)")
    ap.add_argument("--focus", type=int, default=0,
                    help="자동초점을 끌 때 고정할 초점값 (기본 0 = C920 무한대 쪽)")
    ap.add_argument("--pos", default="40,40", metavar="X,Y", help="창 왼쪽 위 위치 (기본 40,40)")
    ap.add_argument("--out-dir", default="logs/cam_raw", help="저장 폴더 (기본 logs/cam_raw)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        req_w, req_h = (int(v) for v in args.size.lower().split("x"))
        win_x, win_y = (int(v) for v in args.pos.split(","))
    except ValueError:
        log.error("--size는 WxH (예: 1920x1080), --pos는 X,Y (예: 40,40) 형식")
        return 1
    # Qt는 디스플레이가 없으면 예외 없이 abort한다 → 먼저 확인
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log.error("DISPLAY 없음 — 창을 띄울 수 없음 (예: DISPLAY=:1)")
        return 1

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        log.error("카메라 열기 실패: /dev/video%d", args.device)
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("카메라 열림: /dev/video%d 요청 %dx%d → 실제 %dx%d @%.0f (MJPG)",
             args.device, req_w, req_h, w, h, cap.get(cv2.CAP_PROP_FPS))

    ctrl = V4L2Ctrl(f"/dev/video{args.device}")
    focus = Focus(ctrl)
    if focus.controllable:
        if args.autofocus == "on":
            focus.set_auto(True)
        else:
            focus.lock(args.focus)
        lo, hi, step = focus.range or (0, 0, 0)
        log.info("초점: %s (초점값 범위 %d~%d, 단계 %d)", focus.label(), lo, hi, step)
    else:
        log.info("초점: 자동초점·초점 컨트롤 없음 — 고정초점 카메라, 설정 안 바꿈")

    base = f"{WIN} {w}x{h} /dev/video{args.device}"
    win = base  # 창 식별자. 표시 제목은 setWindowTitle로 바꾼다
    # AUTOSIZE = 화면 픽셀 1:1, 창 크기 조절 없음. GUI_NORMAL = Qt 툴바 없음
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    def show_title() -> None:
        keys = "q quit, s save" + (", a AF, [ ] focus" if focus.controllable else "")
        cv2.setWindowTitle(win, f"{base}  [{focus.label()}]  ({keys})")

    out_dir = REPO / args.out_dir
    placed = False
    misses = n = 0
    t0 = time.perf_counter()
    log.info("시작 — 키: q 종료 / s 원본 저장" + (" / a 자동초점 / [ ] 초점값" if focus.controllable else ""))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses % 30 == 1:
                    log.warning("카메라 프레임 없음 (%d회 연속)", misses)
                if misses > MAX_MISSES:
                    log.error("카메라 응답 없음 — 종료")
                    break
                cv2.waitKey(10)
                continue
            misses = 0
            n += 1

            cv2.imshow(win, frame)  # 받은 프레임 그대로
            if not placed:
                cv2.moveWindow(win, win_x, win_y)
                show_title()
                placed = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                # 프레임 번호를 넣어 같은 초에 여러 번 눌러도 덮어쓰지 않게
                p = out_dir / f"raw_{time.strftime('%Y%m%d_%H%M%S')}_{n:06d}_{w}x{h}.png"
                cv2.imwrite(str(p), frame)
                log.info("저장: %s (%s)", p, focus.label())
            elif key == ord("a") and focus.has_auto:
                if focus.auto():
                    focus.lock(focus.value() or 0)  # 끄면서 렌즈를 보고된 값에 확실히 고정
                else:
                    focus.set_auto(True)
                log.info("초점: %s", focus.label())
                show_title()
            elif key in (ord("["), ord("]")) and focus.range:
                if focus.auto():
                    log.info("자동초점이 켜져 있음 — a로 끈 뒤 조절")
                else:
                    sign = 1 if key == ord("]") else -1
                    focus.set_value((focus.value() or 0) + sign * focus.range[2])
                    log.info("초점: %s", focus.label())
                    show_title()
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break  # 창 닫기 버튼
            if n % 300 == 0:
                log.info("수신 %.1f fps", n / (time.perf_counter() - t0))
    except KeyboardInterrupt:
        log.info("Ctrl+C — 종료")
    finally:
        cap.release()
        ctrl.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
