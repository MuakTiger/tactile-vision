#!/usr/bin/env python3
"""
핀 구동 지연 측정 — 화면 흰·검 전환과 칸 하나의 핀을 폰 슬로모션(240 fps)으로 같이 찍는다.

젯슨이 깊이 계산 없이 칸 하나(기본 9)의 단계를 lo ↔ hi(기본 0 ↔ 7)로 바꿔 보낸다.
  - 평소 장치처럼 초당 15번 계속 보낸다. Pico(펌웨어 v3)는 0.2 s 박자마다 그때 받아 둔 최신 값만 적용한다.
  - 단계를 바꾸는 프레임을 보낸 바로 뒤에 창 전체를 흰색(hi)·검은색(lo)으로 바꾼다 → 영상의 0 ms.
  - 바꾸는 간격은 --hold-min ~ --hold-max 초에서 무작위(박자와 어긋나는 위치가 골고루 섞이게).
재는 구간: 전송 + 박자 대기 + 서보 이동(+ 모니터 표시 지연만큼 짧게 나옴). 젯슨 계산 시간은 들어가지 않는다
(로그의 처리 시간을 따로 더한다).

  python tools/servo_latency_flip.py --count 10      # 오르기 10번 + 내리기 10번 (녹화 1개, 약 1분)
  python tools/servo_latency_flip.py --no-hw         # Pico 없이 화면만 (점검용, 핀 안 움직임)

기록: logs/latency_flip_<월일_시분>.csv — 바꾼 순간마다 n, dir, level, t_send_s(시작부터), t_shown_s, show_ms
키: q / ESC 중단 (핀은 천천히 홈으로)
설정(config/default.yaml)과 캘리브레이션은 읽기만 한다. run.py와 logs/metrics.csv는 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision.calibration import CalibTable  # noqa: E402
from tactile_vision.config import load  # noqa: E402
from tactile_vision.pipeline import Pipeline  # noqa: E402
from tactile_vision.transport import PicoLink  # noqa: E402

WIN = "latency flip"
log = logging.getLogger("latency_flip")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cell", type=int, default=9, help="움직일 칸 (0~23, 기본 9 = 위에서 둘째 줄·왼쪽에서 넷째)")
    ap.add_argument("--lo", type=int, default=0, help="낮은 단계 (검은 화면)")
    ap.add_argument("--hi", type=int, default=7, help="높은 단계 (흰 화면)")
    ap.add_argument("--count", type=int, default=10, help="오르기 횟수 (내리기도 같은 수)")
    ap.add_argument("--hold-min", type=float, default=2.5)
    ap.add_argument("--hold-max", type=float, default=3.5)
    ap.add_argument("--rate", type=float, default=15.0, help="전송 횟수/초 (평소 장치와 비슷하게)")
    ap.add_argument("--seed", type=int, default=None, help="간격 난수 시드 (기본: 시각)")
    ap.add_argument("--size", default="960x720", help="창 크기")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-hw", action="store_true", help="Pico 없이 화면만")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load(a.config)
    for w in cfg.validate():
        log.debug("설정 주의: %s", w)
    n_lv = cfg.mapping.levels
    if not (0 <= a.cell < geo.CELL_COUNT and 0 <= a.lo < a.hi < n_lv):
        ap.error(f"cell 0~{geo.CELL_COUNT - 1}, 0 <= lo < hi < {n_lv}")
    rng = random.Random(a.seed)
    W, H = (int(v) for v in a.size.lower().split("x"))

    link = None
    if not a.no_hw:
        link = PicoLink(cfg.transport, required=True)
        calib = CalibTable.load_or_default(cfg.path(cfg.calibration.file), levels=n_lv)
        # run.py와 같은 기동 동기화(박자·진동 설정, 캘리브레이션, 홈 복귀)
        Pipeline.sync_hardware(SimpleNamespace(cfg=cfg, link=link, calib=calib))

    out = REPO / "logs" / f"latency_flip_{time.strftime('%m%d_%H%M')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("x", newline="", encoding="utf-8")
    wr = csv.writer(fh)
    tick_ms = round(1000.0 / cfg.transport.pin_rate_hz)
    fh.write(f"# cell={a.cell} lo={a.lo} hi={a.hi} rate={a.rate:g}Hz pin_tick={tick_ms}ms "
             f"hold={a.hold_min:g}~{a.hold_max:g}s hw={'no' if a.no_hw else 'pico'} "
             f"fw={getattr(link, 'fw_version', None)}\n")
    wr.writerow(["n", "dir", "level", "t_send_s", "t_shown_s", "show_ms"])

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    if a.fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    frames = {v: np.full((H, W, 3), 255 if v else 0, np.uint8) for v in (0, 1)}

    def show(white: bool, note: str) -> None:
        img = frames[int(white)].copy()
        # 작은 글씨만 구석에. 화면 대부분은 순수한 흰·검으로 둔다
        cv2.putText(img, note, (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (120, 120, 120), 1, cv2.LINE_AA)
        cv2.imshow(WIN, img)
        cv2.waitKey(1)

    levels = np.zeros(geo.CELL_COUNT, np.int32)
    hazards = np.zeros(geo.CELL_COUNT, np.int32)
    levels[a.cell] = a.lo
    period = 1.0 / a.rate
    t0 = time.perf_counter()
    total = 2 * a.count
    n = 0
    high = False
    show(False, f"ready  0/{total}")
    next_switch = t0 + 3.0  # 처음 3초는 낮은 단계(검은 화면)
    next_tx = t0
    last_drain = last_ping = t0
    log.info("시작: 칸 %d, %d ↔ %d단계, %d번 전환, 기록 %s", a.cell, a.lo, a.hi, total, out)
    try:
        while n < total or time.perf_counter() < next_switch:
            now = time.perf_counter()
            if n < total and now >= next_switch:
                high = not high
                levels[a.cell] = a.hi if high else a.lo
                t_send = time.perf_counter()
                if link is not None:
                    link.send_grid(levels, hazards, force=True)
                show(high, f"{'UP' if high else 'DOWN'}  {n + 1}/{total}")
                t_shown = time.perf_counter()
                n += 1
                wr.writerow([n, "up" if high else "down", int(levels[a.cell]), f"{t_send - t0:.4f}",
                             f"{t_shown - t0:.4f}", f"{(t_shown - t_send) * 1000:.2f}"])
                fh.flush()
                next_switch = t_send + rng.uniform(a.hold_min, a.hold_max)
                next_tx = t_send + period
            elif now >= next_tx:
                if link is not None:
                    link.send_grid(levels, hazards, force=True)
                next_tx += period
                if now - next_tx > period:  # 밀렸으면 다시 맞춤
                    next_tx = now + period
            if link is not None and now - last_drain >= 2.0:
                last_drain = now
                if now - last_ping >= 10.0:
                    last_ping = now
                    link.send_ping()
                for line in link.drain():
                    log.info("Pico: %s", line)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                log.info("중단 (전환 %d/%d)", n, total)
                break
            time.sleep(0.001)
    finally:
        fh.close()
        if link is not None:
            link.send_ping()
            time.sleep(0.2)
            for line in link.drain():
                log.info("Pico: %s", line)
            link.close(home_first=True)
        cv2.destroyAllWindows()
    log.info("끝: 전환 %d번 기록 → %s", n, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
