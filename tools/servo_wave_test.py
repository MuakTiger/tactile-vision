#!/usr/bin/env python3
"""
서보 파도 시험 — 모터만 움직인다 (추론·카메라 없음).

24개 핀이 정해진 순서대로 하나씩 천천히 올라갔다 내려간다. 앞 핀이 내려가는 동안 다음 핀이 올라오고,
마지막 핀 다음에는 쉬지 않고 첫 핀부터 다시 이어져 파도가 끊기지 않는다.
경로: 젯슨 → USB → Pico 펌웨어 v3 → (칸→채널 표) → PCA9685 ×2 → SG90.

순서 (--order)
  channel  PCA 채널 순서 (기본): 0x40 보드 채널 0→11, 이어서 0x41 보드 채널 0→11.
           패드 위에서는 왼쪽 위부터 세로 2칸씩(위→아래) 오른쪽으로 지그재그, 위 두 줄 다음 아래 두 줄.
           칸 번호로: 0 6 1 7 2 8 3 9 4 10 5 11 | 12 18 13 19 14 20 15 21 16 22 17 23
  cell     칸 번호 순서 0→23 (화면 순서: 한 줄씩 왼쪽→오른쪽, 위 줄부터)
  column   열 단위 왼쪽→오른쪽 (한 열의 4칸이 함께)

움직임 (--mode)
  smooth   기본. 서보 위치를 1 tick(≈0.46°) 단위로 50 Hz 갱신 → 계단 없이 연속으로 움직인다.
           캘리브레이션용 직접 지령(RAW_TICK)을 쓴다 — 펌웨어의 박자·속도 제한을 거치지 않는다.
  levels   실제 장치 경로(GRID): 높이 16단계, 이 시험 동안만 Pico 박자 40 ms. 단계 사이가 조금 계단처럼 보인다.

  python tools/servo_wave_test.py                          채널 순서, 파도 3번 (한 번 12 s)
  python tools/servo_wave_test.py --loops 0                Ctrl+C 할 때까지 계속
  python tools/servo_wave_test.py --bump 6 --gap 0.75      더 천천히 (핀 하나 오르내림 6 s, 다음 핀 0.75 s 뒤)
  python tools/servo_wave_test.py --order cell
  python tools/servo_wave_test.py --home 372 --span 100    범위(tick): 핀 맨 아래, 위로 올라가는 tick 수

범위 기본값은 calib/servo_calib.json 그대로 — 지금 핀 맨 아래 372 tick(≈124.7°) ↔ 맨 위 232 tick(≈60°), 역방향 조립.
--home/--span을 주면 모든 칸 같은 범위, 올라가는 방향은 파일을 따른다.
배선이 맞으면 핀이 위 순서대로 움직인다. 순서가 어긋나는 핀은 서보 커넥터가 다른 채널에 꽂힌 것이다.
캘리브레이션 파일은 건드리지 않는다(범위는 Pico 메모리에만). 끝나거나 Ctrl+C면 펌웨어가 전체를 천천히 맨 아래(home)로 내린다.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
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

N = geo.CELL_COUNT
EXPECTED_MAP = "colpairs"  # 펌웨어 tactile_core.h CHANNEL_MAP_NAME
DEFAULT_GAP = {"channel": 0.5, "cell": 0.5, "column": 1.0}
SEND_DT = 0.02  # 50 Hz
_stop = False


def units_for(order: str) -> list[list[int]]:
    """움직이는 순서. 원소 하나(칸 묶음)가 함께 움직인다."""
    if order == "cell":
        return [[c] for c in range(N)]
    if order == "column":
        return [[r * geo.GRID_COLS + col for r in range(geo.GRID_ROWS)] for col in range(geo.GRID_COLS)]
    if order == "channel":
        by_pca = sorted(range(N), key=lambda c: (geo.PCA_ADDRS.index(geo.pca_of(c)[0]), geo.pca_of(c)[1]))
        return [[c] for c in by_pca]
    raise ValueError(order)


class Wave:
    """
    끊기지 않는 파도. 묶음 i는 i·gap에 시작해 bump초 동안 0 → 1 → 0(코사인)으로 오르내리고,
    주기 period = 묶음 수 × gap 마다 반복한다. loops = 0이면 끝없이.
    """

    def __init__(self, units: list[list[int]], gap: float, bump: float, loops: int = 1) -> None:
        if gap <= 0 or bump <= 0 or loops < 0:
            raise ValueError("gap·bump > 0, loops ≥ 0")
        self.units, self.gap, self.bump, self.loops = units, gap, bump, loops
        self.period = len(units) * gap
        if bump > self.period + 1e-9:
            raise ValueError(f"bump {bump:g} s가 한 바퀴 {self.period:g} s보다 길다 — gap을 늘리거나 bump를 줄일 것")
        last = (len(units) - 1) * gap + bump
        self.duration = math.inf if loops == 0 else (loops - 1) * self.period + last

    def start_of(self, i: int, k: int = 0) -> float:
        return k * self.period + i * self.gap

    def frac_at(self, t: float) -> np.ndarray:
        """칸별 높이 0~1."""
        out = np.zeros(N)
        for i, cells in enumerate(self.units):
            u = t - i * self.gap
            if u <= 0:
                continue
            k, r = divmod(u, self.period)
            if self.loops and k >= self.loops:
                continue
            x = r / self.bump
            if x < 1.0:
                out[cells] = 0.5 * (1.0 - math.cos(2.0 * math.pi * x))
        return out

    def levels_at(self, t: float, top: int) -> np.ndarray:
        return np.rint(self.frac_at(t) * top).astype(int)

    def ticks_at(self, t: float, low, high) -> np.ndarray:
        """low·high = 핀 맨 아래·맨 위 tick (숫자 또는 칸별 배열, 역방향 조립이면 high < low)."""
        return np.rint(np.asarray(low) + (np.asarray(high) - np.asarray(low)) * self.frac_at(t)).astype(int)

    def max_raised(self, dt: float = SEND_DT) -> int:
        """한 바퀴 동안 동시에 아래가 아닌 핀의 최대 수 (계획값)."""
        t1 = self.period * (2 if self.loops != 1 else 1)
        return max(int(np.count_nonzero(self.frac_at(k * dt) > 1e-6)) for k in range(int(t1 / dt) + 1))


def label(cells: list[int]) -> str:
    parts = []
    for c in cells:
        addr, ch = geo.pca_of(c)
        r, col = divmod(c, geo.GRID_COLS)
        parts.append(f"칸 {c:2d}({r + 1}행 {col + 1}열, 0x{addr:02X} 채널 {ch:2d})")
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", choices=("channel", "cell", "column"), default="channel",
                    help="움직이는 순서 (기본 channel = PCA 채널 순서)")
    ap.add_argument("--mode", choices=("smooth", "levels"), default="smooth",
                    help="smooth = 1 tick 단위 연속(기본) / levels = 실제 경로 16단계")
    ap.add_argument("--loops", type=int, default=3, help="파도 횟수, 0 = Ctrl+C까지 계속 (기본 3)")
    ap.add_argument("--gap", type=float, default=None,
                    help="다음 핀이 시작하는 간격 s (기본 channel·cell 0.5, column 1.0)")
    ap.add_argument("--bump", type=float, default=4.0, help="핀 하나가 올라갔다 내려오는 시간 s (기본 4.0)")
    ap.add_argument("--home", type=int, default=None, help="핀 맨 아래 tick (기본: 캘리브레이션 파일, 지금 372)")
    ap.add_argument("--span", type=int, default=None, help="위로 올라가는 범위 tick (기본: 캘리브레이션 파일, 지금 140)")
    ap.add_argument("--levels", type=int, default=16, help="levels 모드 단계 수 2~16 (기본 16)")
    ap.add_argument("--beat-ms", type=int, default=40, help="levels 모드 Pico 박자 ms 20~2000 (기본 40)")
    ap.add_argument("--port", default=None, help="Pico 포트 (기본 설정값 /dev/ttyACM0)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    cfg = load()
    levels_mode = args.mode == "levels"
    lv = args.levels if levels_mode else cfg.mapping.levels
    table = CalibTable.for_tool(cfg.path(cfg.calibration.file), lv, args.home, args.span, "servo_wave_test")
    low, high = table.ends()  # 채널별 핀 맨 아래 / 맨 위 tick
    if not (geo.us_to_tick(400) <= min(low.min(), high.min()) and max(low.max(), high.max()) <= geo.us_to_tick(2600)):
        print("범위가 안전 한계 82~533 tick(400~2600 µs) 밖입니다", file=sys.stderr)
        return 1
    if not 2 <= args.levels <= 16 or not 20 <= args.beat_ms <= 2000:
        print("--levels 2~16, --beat-ms 20~2000", file=sys.stderr)
        return 1
    gap = args.gap if args.gap is not None else DEFAULT_GAP[args.order]
    try:
        wave = Wave(units_for(args.order), gap, args.bump, args.loops)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    if args.port:
        cfg.transport.port = args.port
    cfg.transport.max_rate_hz = 0  # 이 도구가 직접 50 Hz로 보낸다
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

    def ping() -> list[str]:
        link.send_ping()
        time.sleep(0.4)
        lines = link.drain(max_lines=400)
        for line in lines:
            if line.startswith(("I2C", "PONG", "ERR", "FAILSAFE")):
                print(f"    Pico: {line}")
        return lines

    print(f"Pico 펌웨어 v{link.fw_version} ({cfg.transport.port})")
    lines = ping()
    m = next((re.search(r"\bmap=(\S+)", ln) for ln in lines if ln.startswith("I2C")), None)
    fw_map = m.group(1) if m else None
    if fw_map != EXPECTED_MAP:
        print(f"  ※ Pico 펌웨어의 칸→채널 배열이 '{fw_map or '알 수 없음(옛 펌웨어)'}' — 새 배열('{EXPECTED_MAP}')이 아니라 "
              "순서가 다르게 보일 수 있습니다. tools/build_firmware.sh --upload 로 올린 뒤 다시 하세요.")
    if any("MISSING" in ln for ln in lines):
        print("  ※ PCA9685 응답 없음 — 그 보드의 칸은 움직이지 않습니다")

    peak_speed = int(np.abs(high - low).max()) * math.pi / args.bump  # tick/s, 코사인 곡선의 가장 빠른 순간
    print(f"범위: {table.range_text()}, 움직임 '{args.mode}'"
          + (f" (1 tick 단위, 50 Hz)" if args.mode == "smooth" else f" ({args.levels}단계, Pico 박자 {args.beat_ms} ms)"))
    print(f"순서 '{args.order}': 간격 {gap:g} s, 핀 하나 오르내림 {args.bump:g} s(가장 빠를 때 {peak_speed / geo.TICKS_PER_DEG:.0f}°/s), "
          f"한 바퀴 {wave.period:g} s, 동시에 올라가 있는 핀 최대 {wave.max_raised()}개, "
          + ("Ctrl+C까지 계속" if args.loops == 0 else f"{args.loops}바퀴 {wave.duration:.1f} s"))
    print("  순서: " + " → ".join(",".join(str(c) for c in u) for u in wave.units))

    link.push_config(levels=lv, max_simultaneous=cfg.safety.max_simultaneous_moves,
                     tick_period_ms=args.beat_ms if levels_mode else int(round(1000 / cfg.transport.pin_rate_hz)),
                     jitter_ticks=[geo.deg_delta_to_ticks(d) for d in cfg.hazard.jitter_deg])
    link.push_calibration(table)
    time.sleep(0.2)
    link.drain(max_lines=400)

    zero = np.zeros(N, dtype=int)

    def send_grid_for(seconds: float) -> None:
        """아래 위치 유지 (GRID 레벨 0 — 펌웨어 속도 제한으로 천천히 이동)."""
        t0 = time.perf_counter()
        nxt = t0
        while not _stop and time.perf_counter() - t0 < seconds:
            link.send_grid(zero, None, force=True)
            nxt += SEND_DT
            time.sleep(max(0.0, nxt - time.perf_counter()))

    sent_tick = low.copy()
    stats = {"raw": 0}

    def run_wave() -> None:
        announced: set[tuple[int, int]] = set()
        t0 = time.perf_counter()
        nxt = t0
        last_tx = t0
        k_shown = -1
        while not _stop:
            now = time.perf_counter()
            t = now - t0
            if t >= wave.duration:
                break
            k = int(t // wave.period)
            in_range = args.loops == 0 or k < args.loops  # 마지막 바퀴 뒤 꼬리(앞 핀들이 내려오는 중)는 표시 안 함
            if k != k_shown and in_range:
                k_shown = k
                print(f"파도 {k + 1}" + ("" if args.loops == 0 else f"/{args.loops}"))
            for i, cells in enumerate(wave.units):
                if in_range and (k, i) not in announced and t >= wave.start_of(i, k):
                    announced.add((k, i))
                    print(f"  {t - k * wave.period:5.2f}s  {label(cells)}")
            if levels_mode:
                link.send_grid(wave.levels_at(t, lv - 1), None, force=True)
                last_tx = now
            else:
                ticks = wave.ticks_at(t, low, high)
                for c in np.nonzero(ticks != sent_tick)[0]:
                    link.send_raw_tick(int(c), int(ticks[c]))
                    sent_tick[c] = ticks[c]
                    stats["raw"] += 1
                    last_tx = now
                if now - last_tx > 0.2:  # 움직이는 핀이 없어도 failsafe(0.5 s)가 걸리지 않게
                    link.send_keepalive()
                    last_tx = now
                link.drain(max_lines=400)  # RAW_TICK마다 오는 "TICK ch t" 응답은 버린다
            nxt += SEND_DT
            time.sleep(max(0.0, nxt - time.perf_counter()))

    try:
        print("맨 아래로...")
        send_grid_for(1.0)
        run_wave()
        send_grid_for(0.3)  # 파도가 끝까지 돌았으면 이미 모두 맨 아래. 중간에 멈췄으면 바로 넘어간다.
    finally:
        if link.connected:
            t = cfg.transport
            link.push_config(levels=cfg.mapping.levels, max_simultaneous=cfg.safety.max_simultaneous_moves,
                             tick_period_ms=int(round(1000 / t.pin_rate_hz)),
                             jitter_ticks=[geo.deg_delta_to_ticks(d) for d in cfg.hazard.jitter_deg])
            print("끝 — Pico 상태:")
            ping()
        link.close(home_first=True)  # 올라가 있던 핀은 펌웨어가 천천히(1 tick / 10 ms) 맨 아래로
    if _stop:
        print("중단됨 (Ctrl+C) — 맨 아래로 천천히 내림")
    print(f"전송 {link.n_sent}회 (직접 지령 {stats['raw']}회), 오류 {link.n_error}회")
    return 0


if __name__ == "__main__":
    sys.exit(main())
