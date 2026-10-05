#!/usr/bin/env python3
"""
전체 모터 느리게 위아래 — 24개가 함께, 천천히, 범위 제한 안에서.

젯슨 → USB → Pico 펌웨어 v3 → PCA9685 ×2 → SG90. 추론·카메라 없음.
위치를 1 tick(≈0.46°) 단위로 50 Hz 갱신해서 계단 없이 부드럽게 움직인다(코사인 곡선: 양 끝에서 더 느리게).

  python tools/servo_slow_updown.py                     한쪽 방향 3 s, 2번 왕복 (약 12 s)
  python tools/servo_slow_updown.py --one-way 8         더 천천히 (한쪽 방향 8 s)
  python tools/servo_slow_updown.py --one-way 1.5       더 빠르게
  python tools/servo_slow_updown.py --span 80           범위를 더 좁게 (위로 80 tick ≈ 37°, 5.2 mm)
  python tools/servo_slow_updown.py --cycles 0          Ctrl+C까지 계속 (멈추는 법: Ctrl+C 또는 pkill -f servo_slow_updown)

범위 제한: 기본은 calib/servo_calib.json의 채널별 범위 그대로 — 지금은 핀 맨 아래 372 tick(≈124.7°) ↔ 맨 위 232 tick(≈60°)
(서보 역방향 조립이라 tick이 작을수록 핀이 위). --home(핀 맨 아래 tick)·--span(위로 올라가는 tick 수)을 주면
모든 칸 같은 범위가 되고, 올라가는 방향은 파일을 따른다.
어떤 경우에도 펌웨어 절대 한계 82~533 tick(400~2600 µs) 안이며, 범위는 최대 200 tick(≈92°).
시작할 때는 펌웨어의 속도 제한으로 맨 아래까지 가고, 끝나거나 Ctrl+C면 천천히 맨 아래로 내려온 뒤 멈춘다.
캘리브레이션 파일은 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision.calibration import CalibTable  # noqa: E402
from tactile_vision.config import load  # noqa: E402
from tactile_vision.transport import PicoLink  # noqa: E402

N = geo.CELL_COUNT
MAX_SPAN = 200          # tick — 이 도구에서 허용하는 최대 범위 (≈92°, 12.9 mm)
DT = 0.02               # 50 Hz
GLIDE_TICKS_PER_S = 60  # 멈출 때 맨 아래로 내려오는 속도 (≈28°/s)
_stop = False


def ramp_frac(t: float, period: float) -> float:
    """t초의 높이 0(맨 아래)~1(맨 위). 아래 → 위 → 아래, 코사인 곡선이라 양 끝에서 속도 0."""
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * (t % period) / period))


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # 파일로 보내도(nohup) 진행 상황이 바로 보이게
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one-way", type=float, default=3.0, help="아래→위 (또는 위→아래) 한 방향 시간 s (기본 3)")
    ap.add_argument("--cycles", type=int, default=2, help="왕복 횟수, 0 = Ctrl+C까지 (기본 2)")
    ap.add_argument("--home", type=int, default=None, help="핀 맨 아래 tick (기본: 캘리브레이션 파일, 지금 372)")
    ap.add_argument("--span", type=int, default=None,
                    help=f"위로 올라가는 범위 tick, 최대 {MAX_SPAN} (기본: 캘리브레이션 파일, 지금 140 ≈ 64.8°, 9.0 mm)")
    ap.add_argument("--port", default=None, help="Pico 포트 (기본 설정값 /dev/ttyACM0)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    cfg = load()
    table = CalibTable.for_tool(cfg.path(cfg.calibration.file), cfg.mapping.levels, args.home, args.span,
                                "servo_slow_updown")
    down, up = table.ends()     # 채널별 핀 맨 아래 / 맨 위 tick
    spans = np.abs(up - down)
    if spans.min() < 2 or spans.max() > MAX_SPAN:
        print(f"범위는 2~{MAX_SPAN} tick (지금 {spans.min()}~{spans.max()})", file=sys.stderr)
        return 1
    if not (geo.us_to_tick(400) <= min(down.min(), up.min()) and max(down.max(), up.max()) <= geo.us_to_tick(2600)):
        print("범위가 안전 한계 82~533 tick(400~2600 µs) 밖입니다", file=sys.stderr)
        return 1
    if args.one_way < 1.0 or args.cycles < 0:
        print("--one-way ≥ 1 s, --cycles ≥ 0", file=sys.stderr)
        return 1
    period = 2.0 * args.one_way

    if args.port:
        cfg.transport.port = args.port
    cfg.transport.max_rate_hz = 0
    try:
        link = PicoLink(cfg.transport, required=True)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    if (link.fw_version or 0) < 3:
        print(f"Pico 펌웨어 v{link.fw_version} — v3가 필요합니다", file=sys.stderr)
        link.close(home_first=False)
        return 1

    def on_signal(signum: int, _frame: object) -> None:
        global _stop
        _stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    peak = spans.max() * math.pi / (2.0 * args.one_way)  # tick/s, 곡선에서 가장 빠른 순간
    print(f"Pico 펌웨어 v{link.fw_version} ({cfg.transport.port})")
    link.send_ping()
    time.sleep(0.4)
    for line in link.drain(max_lines=200):
        if line.startswith("I2C"):
            print(f"  {line}")
    print(f"전체 {N}개 모터: {table.range_text()}")
    print(f"한 방향 {args.one_way:g} s (가장 빠를 때 {peak / geo.TICKS_PER_DEG:.0f}°/s), "
          + ("Ctrl+C까지 계속" if args.cycles == 0 else f"{args.cycles}번 왕복 {args.cycles * period:.0f} s"))

    # 1) 맨 아래로: 펌웨어 보정 범위를 이번 범위로 맞추고 레벨 0 → 펌웨어 속도 제한으로 이동 (급하게 튀지 않게)
    t = cfg.transport
    link.push_config(levels=cfg.mapping.levels, max_simultaneous=cfg.safety.max_simultaneous_moves,
                     tick_period_ms=int(round(1000 / t.pin_rate_hz)),
                     jitter_ticks=[geo.deg_delta_to_ticks(d) for d in cfg.hazard.jitter_deg])
    link.push_calibration(table)
    time.sleep(0.2)
    zero = np.zeros(N, dtype=int)
    print("맨 아래로 이동 중...")
    end = time.perf_counter() + 2.0
    while not _stop and time.perf_counter() < end:
        link.send_grid(zero, None, force=True)
        time.sleep(DT)
    link.drain(max_lines=400)

    # 2) 느린 위아래 — 칸마다 자기 범위 안의 같은 높이를 직접 지령
    cur = down.copy()
    last_tx = time.perf_counter()
    biggest = 0
    err_mark, err_t = link.n_error, time.perf_counter()
    print("시작")
    t0 = time.perf_counter()
    nxt = t0
    k_shown = -1
    while not _stop:
        now = time.perf_counter()
        el = now - t0
        if args.cycles and el >= args.cycles * period:
            break
        if now - err_t >= 1.0:      # 오래 켜 둘 때를 위해: 전송이 계속 실패하면(Pico 분리 등) 스스로 멈춘다
            if link.n_error - err_mark >= 50:
                print("전송 오류가 계속되어 중단합니다 (Pico 연결·케이블 확인)", file=sys.stderr)
                break
            err_mark, err_t = link.n_error, now
        k = int(el // period)
        if k != k_shown:
            k_shown = k
            print(f"  왕복 {k + 1}" + (f"/{args.cycles}" if args.cycles else ""))
        ticks = np.rint(down + (up - down) * ramp_frac(el, period)).astype(int)
        changed = np.nonzero(ticks != cur)[0]
        if changed.size:
            biggest = max(biggest, int(np.abs(ticks - cur).max()))
            for c in changed:
                link.send_raw_tick(int(c), int(ticks[c]))
            cur = ticks
            last_tx = now
        elif now - last_tx > 0.2:   # 양 끝에서 멈춰 있는 동안에도 failsafe(0.5 s)가 걸리지 않게
            link.send_keepalive()
            last_tx = now
        link.drain(max_lines=400)   # RAW_TICK마다 오는 응답 줄은 버린다
        nxt += DT
        time.sleep(max(0.0, nxt - time.perf_counter()))

    # 3) 천천히 맨 아래로 (중간에 멈췄을 때 급하게 떨어지지 않게) — 칸마다 1 tick씩 자기 맨 아래 쪽으로
    if np.any(cur != down):
        print(f"맨 아래로 천천히 (최대 {int(np.abs(cur - down).max())} tick)...")
    while True:
        moving = np.nonzero(cur != down)[0]
        if moving.size == 0:
            break
        cur[moving] += np.sign(down - cur)[moving]
        for c in moving:
            link.send_raw_tick(int(c), int(cur[c]))
        link.drain(max_lines=400)
        time.sleep(1.0 / GLIDE_TICKS_PER_S)
    link.send_ping()
    time.sleep(0.4)
    for line in link.drain(max_lines=400):
        if line.startswith(("I2C", "PONG", "FAILSAFE", "ERR")):
            print(f"  Pico: {line}")
    link.close(home_first=True)  # 보정 home(= 핀 맨 아래)으로 — 펌웨어도 천천히 간다
    if _stop:
        print("중단됨 (Ctrl+C) — 맨 아래로 내려서 멈춤")
    print(f"끝 — 한 번에 최대 {biggest} tick씩 움직임, 전송 {link.n_sent}회, 오류 {link.n_error}회")
    return 0


if __name__ == "__main__":
    sys.exit(main())
