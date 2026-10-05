#!/usr/bin/env python3
"""
서보 배선 점검 — 추론 모델 없이 24칸 전체를 아래 ↔ 위로 움직인다.

실제 장치와 같은 경로를 쓴다: 젯슨 → PicoLink(USB) → Pico 펌웨어 v3 → PCA9685 ×2 → SG90.
캘리브레이션 파일은 건드리지 않는다. 이번 실행 동안만 Pico 메모리에 범위를 넣는다
(Pico를 다시 켜거나 run.py를 실행하면 calib/servo_calib.json 값으로 돌아간다).

  python tools/servo_updown_test.py                    5번 왕복 (아래 1.5 s → 위 1.5 s)
  python tools/servo_updown_test.py --cycles 10 --hold 1.0
  python tools/servo_updown_test.py --home 372 --span 100      범위(tick): 핀 맨 아래, 위로 올라가는 tick 수
  python tools/servo_updown_test.py --hazard 3                 모든 칸에 위험도 3 진동도 같이
  python tools/servo_updown_test.py --channels 0 12            일부 칸만 움직이기 (나머지는 아래)

범위 기본값은 calib/servo_calib.json 그대로 — 지금 핀 맨 아래 372 tick(≈124.7°) ↔ 맨 위 232 tick(≈60°), 역방향 조립.
--home/--span을 주면 모든 칸 같은 범위, 올라가는 방향은 파일을 따른다. 1 tick = 4.88 µs (SG90 명목 500 µs = 0°, 2400 µs = 180°).
끝나면(또는 Ctrl+C) 전체를 home(핀 맨 아래)으로 보낸다 — 펌웨어가 천천히(1 tick / 10 ms) 내린다.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision.calibration import CalibTable  # noqa: E402
from tactile_vision.config import load  # noqa: E402
from tactile_vision.transport import PicoLink  # noqa: E402

LEVELS = 8
SEND_HZ = 10.0  # GRID 전송 빈도 — Pico는 박자(기본 0.2 s)마다 최신 값만 쓴다
_stop = False


def drain(link: PicoLink, quiet_prefix: tuple[str, ...] = ("CFG", "LIMIT")) -> list[str]:
    lines = link.drain(max_lines=80)
    for line in lines:
        if not line.startswith(quiet_prefix):
            print(f"    Pico: {line}")
    return lines


def ping(link: PicoLink) -> list[str]:
    link.send_ping()
    time.sleep(0.4)
    return drain(link)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=5, help="아래→위 왕복 횟수 (기본 5)")
    ap.add_argument("--hold", type=float, default=1.5, help="각 위치에 머무는 시간 s (기본 1.5)")
    ap.add_argument("--home", type=int, default=None, help="핀 맨 아래 tick (기본: 캘리브레이션 파일, 지금 372)")
    ap.add_argument("--span", type=int, default=None, help="위로 올라가는 범위 tick (기본: 캘리브레이션 파일, 지금 140)")
    ap.add_argument("--hazard", type=int, default=0, choices=range(4), help="모든 칸 위험도 진동 0~3 (기본 0)")
    ap.add_argument("--channels", type=int, nargs="+", default=None,
                    help="움직일 칸 번호 0~23 (기본 전체). 나머지 칸은 아래에 둔다")
    ap.add_argument("--port", default=None, help="Pico 포트 (기본 설정값 /dev/ttyACM0)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    cfg = load()
    table = CalibTable.for_tool(cfg.path(cfg.calibration.file), LEVELS, args.home, args.span, "servo_updown_test")
    down_t, up_t = table.ends()
    if not (geo.us_to_tick(400) <= min(down_t.min(), up_t.min()) and max(down_t.max(), up_t.max()) <= geo.us_to_tick(2600)):
        print("범위가 안전 한계 82~533 tick(400~2600 µs) 밖입니다", file=sys.stderr)
        return 1
    moving = sorted(set(args.channels)) if args.channels else list(range(geo.CELL_COUNT))
    if any(not 0 <= c < geo.CELL_COUNT for c in moving):
        print("--channels는 0~23", file=sys.stderr)
        return 1

    if args.port:
        cfg.transport.port = args.port
    cfg.transport.max_rate_hz = 0  # 이 도구가 직접 10 Hz로 보낸다
    try:
        link = PicoLink(cfg.transport, required=True)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    if (link.fw_version or 0) < 3:
        print(f"Pico 펌웨어 v{link.fw_version} — v3가 필요합니다 (tools/build_firmware.sh --upload)", file=sys.stderr)
        link.close(home_first=False)
        return 1

    def on_signal(signum: int, _frame: object) -> None:
        global _stop
        _stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print(f"Pico 펌웨어 v{link.fw_version} ({cfg.transport.port})")
    print(f"범위: {table.range_text()}")
    print(f"움직일 칸: {'전체 24칸' if len(moving) == geo.CELL_COUNT else moving}"
          f"  (칸 0~11 = 1~2행 → 0x40 보드, 칸 12~23 = 3~4행 → 0x41 보드, 보드마다 세로 2칸씩: "
          f"위 줄 채널 0 2 4 6 8 10 / 아래 줄 1 3 5 7 9 11)")
    print(f"왕복 {args.cycles}회, 위치마다 {args.hold:g} s, 위험도 진동 {args.hazard}")
    print("배선 점검:")
    ping(link)

    # 이번 실행 동안만 쓰는 범위 (파일 저장 안 함) — table은 위에서 캘리브레이션 파일 방향대로 만듦
    t = cfg.transport
    link.push_config(levels=LEVELS, max_simultaneous=cfg.safety.max_simultaneous_moves,
                     tick_period_ms=int(round(1000 / t.pin_rate_hz)),
                     jitter_ticks=[geo.deg_delta_to_ticks(d) for d in cfg.hazard.jitter_deg])
    link.push_calibration(table)
    time.sleep(0.2)
    drain(link)

    down = np.zeros(geo.CELL_COUNT, dtype=int)
    up = down.copy()
    up[moving] = LEVELS - 1
    hz = np.zeros(geo.CELL_COUNT, dtype=int)
    hz[moving] = args.hazard

    def hold(levels: np.ndarray, seconds: float) -> None:
        end = time.perf_counter() + seconds
        nxt = time.perf_counter()
        while not _stop and time.perf_counter() < end:
            link.send_grid(levels, hz, force=True)
            nxt += 1.0 / SEND_HZ
            time.sleep(max(0.0, nxt - time.perf_counter()))

    try:
        print("시작 위치(아래)로 이동...")
        hold(down, 1.0)
        for i in range(args.cycles):
            if _stop:
                break
            print(f"  {i + 1}/{args.cycles}  위 ({up_t[moving[0]]} tick)")
            hold(up, args.hold)
            if _stop:
                break
            print(f"  {i + 1}/{args.cycles}  아래 ({down_t[moving[0]]} tick)")
            hold(down, args.hold)
            drain(link)
        print("끝 — Pico 통계와 배선 다시 확인:")
        hold(down, 0.3)
        ping(link)
    finally:
        link.close(home_first=True)  # 전체 home(= 아래 위치)
    if _stop:
        print("중단됨 (Ctrl+C) — home으로 보냄")
    print(f"전송 {link.n_sent}회, 오류 {link.n_error}회")
    return 0


if __name__ == "__main__":
    sys.exit(main())
