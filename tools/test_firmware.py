"""
Pico 펌웨어 v3 동작 검증 — 실기 펌웨어와 같은 코드(firmware/pico/tactile_pico/tactile_core.h)를
PC에서 컴파일해 가상 시간(1 ms 간격)으로 돌린다. 프레임은 전부 Python(protocol.py)이 만든다.

  1. 박자 적용: 15 Hz로 보낸 값 중 박자(200 ms) 순간의 최신 값만, 박자 시각에만 적용
  2. 진동: 박자의 2·4번째 1/4 구간, 위험도 1/2/3 → 2/4/6 tick, 홈 쪽(맨 아래 핀은 위로), 범위 안
  3. 즉시 처리: ESTOP·HOME_ALL은 박자를 기다리지 않음
  4. failsafe: 500 ms 무통신 → 홈(슬루 제한), 재수신 시 해제 / KEEPALIVE는 failsafe만 막음
  5. RAW_TICK(캘리브레이션): 즉시, 박자가 덮어쓰지 않음
  6. SET_CONFIG: v3(박자·진동 tick) 즉시 반영, v2 4바이트 형식 호환, 잘못된 길이 거부
  7. 슬루·동시 구동 제한, Pico I2C 부담 추정
  8. Python ↔ 펌웨어 일치: 레벨→tick, CRC·파싱, PicoLink(가짜 시리얼)가 보낸 설정을 펌웨어가 받아들임

  python tools/test_firmware.py
"""

from __future__ import annotations

import math
import random
import re
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision import protocol as P  # noqa: E402
from tactile_vision import transport as T  # noqa: E402
from tactile_vision.calibration import ChannelCalib  # noqa: E402
from tactile_vision.config import TransportCfg  # noqa: E402

FAIL = 0
N = P.CELL_COUNT
# 펌웨어 기동 기본값 (tactile_core.h HOME/MAX_TICK_DEFAULT) — 2026-10-05 이 장치: 역방향 조립, 232~372 tick
HOME, MAX = 372, 232
LO, HI = min(HOME, MAX), max(HOME, MAX)
LV = [ChannelCalib(0, HOME, MAX).tick_for_level(lv, 8) for lv in range(8)]  # 레벨 0(먼 곳) → 7(가까운 곳)
TO_HOME = 1 if HOME > MAX else -1


def jit_exp(base: int, a: int) -> int:
    """펌웨어 jitterTarget과 같은 규칙: 홈 쪽으로 a tick, 범위를 벗어나면 반대쪽, 범위 안으로."""
    if a == 0:
        return base
    w = base + TO_HOME * a
    if not LO <= w <= HI:
        w = base - TO_HOME * a
    return min(HI, max(LO, w))


def check(cond: bool, msg: str) -> None:
    global FAIL
    if cond:
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


# ── 하네스 빌드 ──
tmp = tempfile.TemporaryDirectory(prefix="tv_fw_")
EXE = Path(tmp.name) / "fw_test"
r = subprocess.run(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-o", str(EXE),
                    str(REPO / "firmware/pico/test_harness.cpp")], capture_output=True, text=True)
print("=" * 72)
print("0. 하네스 빌드 (실기와 같은 tactile_core.h)")
print("=" * 72)
check(r.returncode == 0, "g++ -std=c++17 -Wall -Wextra 빌드 성공")
check(r.stderr.strip() == "", "컴파일 경고 없음" + (f": {r.stderr.strip()[:300]}" if r.stderr.strip() else ""))
if r.returncode != 0:
    sys.exit(1)


class Sim:
    """시뮬레이션 결과. writes[ch] = [(t, tick)], targets[ch] = [(t, tick)], says = [(t, msg)]."""

    def __init__(self, events: list[tuple[int, bytes]], end_ms: int) -> None:
        blob = b"".join(struct.pack("<IH", t, len(b)) + b for t, b in sorted(events, key=lambda e: e[0]))
        out = subprocess.run([str(EXE), "--sim", str(end_ms)], input=blob, capture_output=True,
                             timeout=120)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.decode())
        self.writes: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.targets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.says: list[tuple[int, str]] = []
        self.stats: dict[str, int] = {}
        for line in out.stdout.decode().splitlines():
            parts = line.split(" ", 2)
            if parts[0] == "W":
                t, ch, tick = map(int, line.split()[1:])
                self.writes[ch].append((t, tick))
            elif parts[0] == "T":
                t, ch, tick = map(int, line.split()[1:])
                self.targets[ch].append((t, tick))
            elif parts[0] == "S":
                self.says.append((int(parts[1]), parts[2]))
            elif parts[0] == "STATS":
                self.stats = {k: int(v) for k, v in (kv.split("=") for kv in line.split()[1:])}

    def tick_at(self, ch: int, t: int) -> int:
        """t 시각에 명령 중인 위치 (마지막 W ≤ t)."""
        v = None
        for tw, tick in self.writes[ch]:
            if tw > t:
                break
            v = tick
        return v

    def said(self, prefix: str) -> list[tuple[int, str]]:
        return [(t, m) for t, m in self.says if m.startswith(prefix)]


def grid(levels: dict[int, int] | None = None, hazards: dict[int, int] | None = None, seq: int = 0,
         flags: int = P.FLAG_NONE) -> bytes:
    lv = np.zeros(N, dtype=int)
    hz = np.zeros(N, dtype=int)
    for k, v in (levels or {}).items():
        lv[k] = v
    for k, v in (hazards or {}).items():
        hz[k] = v
    return P.grid_frame(lv, hz, seq, flags)


def stream(t0: int, t1: int, period: int, make) -> list[tuple[int, bytes]]:
    """t0부터 period 간격으로 t1까지 make(k) 프레임."""
    return [(t0 + period * k, make(k)) for k in range((t1 - t0) // period + 1)]


# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("1. 박자 적용 — 젯슨 15 Hz(66 ms), Pico 박자 200 ms")
print("=" * 72)
ev = stream(10, 2000, 66, lambda k: grid({0: k % 8, 1: (3 * k) % 8}, seq=k))
s = Sim(ev, 2400)
frame_t = [t for t, _ in ev]
beats = list(range(200, 2001, 200))
want0 = {}
for b in beats:
    k = max(i for i, t in enumerate(frame_t) if t <= b)
    want0[b] = (LV[k % 8], LV[(3 * k) % 8])
ok_time = all(t % 200 == 0 for ch in range(N) for t, _ in s.targets[ch])
check(ok_time, "목표 변화는 모두 박자 시각(200 ms 배수)에만")
cur = [HOME, HOME]
ok_val = True
for b in beats:
    for ch in (0, 1):
        at_b = [v for t, v in s.targets[ch] if t == b]
        if at_b:
            cur[ch] = at_b[-1]
        if cur[ch] != want0[b][ch]:
            ok_val = False
check(ok_val, "박자마다 그 순간의 최신 값이 적용됨 (박자 10번 × 2채널)")
st = s.stats
check(st.get("frames") == len(ev) and st.get("applied") == len(beats),
      f"수신 {st.get('frames')}/{len(ev)}, 적용 {st.get('applied')}/{len(beats)}회")
check(st.get("sup") == len(ev) - len(beats),
      f"적용 전에 덮어써진 값 {st.get('sup')}개 = 수신 − 적용 ({len(ev) - len(beats)}) — 버퍼에 쌓이지 않음")
steps = [abs(b - a) for ch in (0, 1) for (_, a), (_, b) in zip(s.writes[ch][1:], s.writes[ch][2:])]
check(max(steps) <= 4, f"슬루 제한: 한 번에 최대 {max(steps)} tick (≤ 4 / 5 ms)")
gaps = [b - a for ch in (0, 1) for (a, _), (b, _) in zip(s.writes[ch][1:], s.writes[ch][2:])]
check(min(gaps) >= 5, f"같은 채널 쓰기 간격 최소 {min(gaps)} ms (서보 갱신 5 ms)")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("2. 위험도 진동 — 박자의 2·4번째 1/4 구간, 1/2/3 → 2/4/6 tick")
print("=" * 72)
LVS = {0: 3, 1: 3, 2: 3, 3: 0, 4: 7, 5: 3}
HZS = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3}
JIT = {0: 0, 1: 2, 2: 4, 3: 6}
s = Sim(stream(10, 3000, 66, lambda k: grid(LVS, HZS, seq=k)), 3000)
exp = {ch: (LV[LVS[ch]], jit_exp(LV[LVS[ch]], JIT[HZS.get(ch, 0)])) for ch in LVS}
exp[6] = (HOME, HOME)
desc = {0: "위험도1 → 2 tick 홈 쪽", 1: "위험도2 → 4 tick", 2: "위험도3 → 6 tick",
        3: "맨 아래(홈) 핀 위험도3 → 위로 6 tick", 4: "맨 위 핀 위험도3 → 아래로 6 tick",
        5: "위험도 0 → 진동 없음", 6: "빈 셀 → 그대로"}
check(exp[3][1] == HOME - TO_HOME * 6 and exp[4][1] == MAX + TO_HOME * 6,
      f"기대값 점검: 맨 아래 핀은 위로({HOME}→{exp[3][1]}), 맨 위 핀은 아래로({MAX}→{exp[4][1]})")
for ch, (base, off) in exp.items():
    ok = True
    for b in range(1000, 3000, 200):
        for dt, want in ((20, base), (70, off), (120, base), (170, off)):
            if s.tick_at(ch, b + dt) != want:
                ok = False
    check(ok, f"ch{ch} {desc[ch]}: 1/4 구간 0·2 = {base}, 1·3 = {off} (박자 10번)")
b2, o2 = exp[2]
starts = [t for (_t0, a), (t, b) in zip(s.writes[2], s.writes[2][1:]) if a == b2 and b != b2 and 1000 <= t < 3000]
check(len(starts) == 20, f"진동 횟수 2 s에 {len(starts)}회 = 초당 {len(starts) / 2:.0f}회 (박자 5 Hz의 2배)")
check(all((t - 1000) % 100 == 50 for t in starts), "진동 시작 시각 = 박자 + 50 / + 150 ms")
inside = all(LO <= v <= HI for ch in range(N) for _, v in s.writes[ch])
check(inside, f"모든 채널이 보정 범위 [{LO}, {HI}] 밖으로 나가지 않음")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("3. ESTOP은 즉시, HOME_ALL(종료)은 천천히")
print("=" * 72)
ev = stream(10, 1490, 66, lambda k: grid({0: 5}, seq=k))
ev.append((1500, grid({0: 5}, seq=100, flags=P.FLAG_ESTOP)))
ev += stream(1566, 1990, 66, lambda k: grid({0: 5}, seq=101 + k))
ev.append((2010, P.home_all_frame(150)))
s = Sim(ev, 3400)
check(s.tick_at(0, 1499) == LV[5] and s.tick_at(0, 1500) == HOME,
      f"ESTOP 순간(1500 ms) 바로 홈: {s.tick_at(0, 1499)} → {s.tick_at(0, 1500)} (슬루 없이, 박자 기다리지 않음)")
check([t for t, _ in s.said("ESTOP")] == [1500], "ESTOP 알림 1회")
held = [(t, v) for t, v in s.targets[0] if 1500 < t < 1600]
check(not held and [v for t, v in s.targets[0] if t == 1600] == [LV[5]],
      "ESTOP 뒤 새 값은 다음 박자(1600 ms)에 적용")
ret = [(t, v) for t, v in s.writes[0] if t >= 2010]
dist = abs(LV[5] - HOME)
arrive = next((t for t, v in ret if v == HOME), None)
check(s.said("HOMED") and s.said("HOMED")[0][0] == 2010 and s.tick_at(0, 2009) == LV[5]
      and abs(s.tick_at(0, 2015) - LV[5]) <= 1, "HOME_ALL(2010 ms)은 점프하지 않음 (시작 직후 1 tick 이하)")
steps = [abs(b - a) for (_, a), (_, b) in zip(ret, ret[1:])]
gaps = [b - a for (a, _), (b, _) in zip(ret, ret[1:])]
check(arrive is not None and max(steps) == 1 and min(gaps) >= 10,
      f"천천히 맨 아래로: {LV[5]} → {HOME} tick, 1 tick씩 10 ms 간격, {dist} tick을 {arrive - 2010 if arrive else '-'} ms "
      f"(≈{1000 / 10 / 2.1619:.0f}°/s)")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("4. failsafe(통신 끊김)는 천천히, KEEPALIVE")
print("=" * 72)
ev = stream(10, 990, 66, lambda k: grid({0: 7}, seq=k))
last = max(t for t, _ in ev)
ev += stream(3200, 3600, 66, lambda k: grid({0: 2}, seq=50 + k))
s = Sim(ev, 3800)
fs = s.said("FAILSAFE")
check([t for t, _ in fs] == [last + 501], f"마지막 수신 {last} ms + 500 ms 뒤 FAILSAFE ({[t for t, _ in fs]})")
down = [(t, v) for t, v in s.writes[0] if last + 501 <= t < 3200]
dv = [v for _, v in down]
dur = (next(t for t, v in down if v == HOME) - (last + 501)) if HOME in dv else None
check(dv and dv[-1] == HOME and max(abs(b - a) for a, b in zip(dv, dv[1:])) == 1,
      f"failsafe 홈 복귀는 1 tick씩 천천히 ({MAX} → {dv[-1] if dv else '-'}, {abs(MAX - HOME)} tick을 {dur} ms)")
check([t for t, _ in s.said("COMM ok")] == [3200], "재수신(3200 ms) 시 failsafe 해제")
check(any(v == LV[2] and 3200 <= t <= 3400 for t, v in s.targets[0]), "해제 후 다음 박자에 새 값 적용")
back = [v for t, v in s.writes[0] if t >= 3200]
check(len(back) > 1 and max(abs(b - a) for a, b in zip(back, back[1:])) == 4,
      "해제 후에는 보통 속도(5 ms당 4 tick)로 움직임")
check(s.stats.get("fs") == 1, "failsafe 1회")

ev = [(10, grid({0: 4}))] + [(t, P.keepalive_frame(i)) for i, t in enumerate(range(200, 3001, 200))]
s = Sim(ev, 3000)
check(not s.said("FAILSAFE"), "KEEPALIVE 0.2 s마다 → 3 s 동안 failsafe 없음")
check(not s.says, "KEEPALIVE는 응답 줄을 만들지 않음")
check(s.targets[0] == [(200, LV[4])], "KEEPALIVE는 핀 값을 바꾸지 않음")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("5. RAW_TICK (캘리브레이션)")
print("=" * 72)
ev = [(10, grid({}))] + [(t, P.keepalive_frame(i)) for i, t in enumerate(range(200, 2001, 200))]
ev += [(500, P.raw_tick_frame(7, 300, 90)), (1500, grid({}, seq=91))]
s = Sim(ev, 2000)
check(s.tick_at(7, 499) == HOME and s.tick_at(7, 500) == 300, "RAW_TICK 즉시 적용 (500 ms, 슬루 없이)")
check([v for t, v in s.targets[7] if 500 < t < 1500] == [], "그 뒤 박자들이 캘리브레이션 값을 덮어쓰지 않음")
check([v for t, v in s.targets[7] if t == 1600] == [HOME], "GRID가 다시 오면 다음 박자(1600 ms)부터 정상 동작")
check(s.said("TICK 7 300"), "TICK 응답")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("6. SET_CONFIG — 박자·진동 진폭 변경, v2 형식, 잘못된 길이")
print("=" * 72)
ev = stream(10, 2990, 50, lambda k: grid({0: 1 + (k // 2) % 3, 1: 3}, {1: 1}, seq=k % 256))
ev += [(1000, P.config_frame(8, 8, 100, (3, 5, 7), 200)),
       (2000, P.config_frame_legacy(8, 8, 75, 201)),
       (2100, P.build_frame(P.TYPE_SET_CONFIG, b"\x08\x08\x00\x64\x03", 202))]
s = Sim(ev, 3000)
cfg = s.said("CFG")
check(cfg and cfg[0] == (1000, "CFG lv=8 sim=8 tick=100 jit=3,5,7"), f"v3 설정 응답: {cfg[0] if cfg else '-'}")
before = [t for t, _ in s.targets[0] if t < 1000]
after = [t for t, _ in s.targets[0] if t > 1000]
check(before and all(t % 200 == 0 for t in before) and after and all((t - 1000) % 100 == 0 for t in after),
      f"박자 200 → 100 ms: 이후 목표 변화 {len(after)}번 모두 100 ms 간격")
ok = all(s.tick_at(1, b + 12) == LV[3] and s.tick_at(1, b + 37) == jit_exp(LV[3], 3)
         and s.tick_at(1, b + 87) == jit_exp(LV[3], 3) for b in range(1500, 2000, 100))
check(ok, "진폭 변경 반영: 위험도1 → 3 tick, 100 ms 박자의 25~50·75~100 ms 구간")
check(len(cfg) >= 2 and cfg[1] == (2000, "CFG lv=8 sim=8 tick=100 jit=3,5,7 (v2 cfg)"),
      "v2 4바이트 설정도 받음 (박자·진폭은 그대로)")
check(s.said("ERR cfg_len=5"), "잘못된 길이(5바이트) 거부")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("7. 동시 구동 제한과 I2C 부담 (최악 조건)")
print("=" * 72)
ev = stream(10, 3000, 100, lambda k: grid({i: (7 if (k // 4) % 2 == 0 else 0) for i in range(N)},
                                          {i: 3 for i in range(0, N, 4)}, seq=k))
s = Sim(ev, 3000)
per_t = defaultdict(int)
for ch in range(N):
    for t, _ in s.writes[ch]:
        if t > 0:
            per_t[t] += 1
check(max(per_t.values()) <= 8, f"5 ms마다 움직이는 서보 최대 {max(per_t.values())}개 (≤ 동시 구동 8)")
n_w = sum(per_t.values())
busy = n_w / 3.0 * 135e-6  # PCA9685 setPWM 1회 약 135 µs @400 kHz (주소+레지스터+4바이트, 산정치)
print(f"        24개 핀이 0.4 s마다 끝↔끝 + 6칸 진동: 쓰기 {n_w / 3.0:.0f}회/초 → I2C 점유 약 {busy * 100:.0f}% (산정치)")
check(busy < 0.5, "I2C 점유 50% 미만 — Pico 루프 여유 있음")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("8. Python ↔ 펌웨어 일치")
print("=" * 72)
rng = random.Random(3)
mism = 0
for _ in range(150):
    home, mx = rng.randint(82, 533), rng.randint(82, 533)
    levels = rng.randint(2, 16)
    fw = subprocess.run([str(EXE), "--ticks", str(home), str(mx), str(levels)], capture_output=True,
                        text=True).stdout.split()
    py = [ChannelCalib(0, home, mx).tick_for_level(lv, levels) for lv in range(levels)]
    mism += [int(v) for v in fw] != py
check(mism == 0, f"레벨→tick 150개 조합(역방향 장착 포함) Python = 펌웨어 (불일치 {mism})")

rng2 = np.random.default_rng(0)
lv_hz = [(rng2.integers(0, 8, N), rng2.integers(0, 4, N)) for _ in range(5)]
blob = b"".join(P.grid_frame(a, b, i) for i, (a, b) in enumerate(lv_hz))
bad = bytearray(P.grid_frame(np.full(N, 7), None, 9))
bad[8] ^= 1
blob += bytes(bad) + b"\x13\xaa\x00\xaa" + P.config_frame(8, 8, 200, (2, 4, 6), 10) + P.keepalive_frame(11)
out = subprocess.run([str(EXE)], input=blob, capture_output=True).stdout.decode()
pls = [ln.split("payload=")[1] for ln in out.splitlines() if ln.startswith("FRAME type=1")]
check(pls == [P.pack_cells(a, b).hex() + "00" for a, b in lv_hz],
      "GRID 5개 내용이 Python과 같음, CRC 손상 1개 거부")
check("STATS frames=7 crc_err=1" in out, "SET_CONFIG·KEEPALIVE 포함 7개 수신, CRC 오류 1개")


class FakeSerial:
    """PicoLink용 가짜 시리얼. readline은 정해 둔 줄을 한 번씩 돌려주고 끝나면 빈 줄."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = list(lines)
        self.out = bytearray()
        self.is_open = True
        self.in_waiting = 0

    def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""

    def write(self, b: bytes) -> int:
        self.out += b
        return len(b)

    def close(self) -> None:
        self.is_open = False


def link_with(lines: list[bytes]) -> tuple[T.PicoLink, FakeSerial]:
    fake = FakeSerial(lines)
    orig = T.serial.Serial
    T.serial.Serial = lambda *a, **k: fake  # noqa: E731
    try:
        return T.PicoLink(TransportCfg()), fake
    finally:
        T.serial.Serial = orig


for label, lines, want_v, want_cfg in (
    ("v3", [b"PICO_READY v3 cells=24 proto=bin-crc8 tick=200 jit=2,4,6\n"], 3,
     ["CFG lv=8 sim=8 tick=100 jit=2,4,6"]),
    ("v2", [b"PONG f=1 crc=0 resync=0 drop=0\n"], 2, ["CFG lv=8 sim=8 tick=200 jit=2,4,6 (v2 cfg)"]),
    ("응답 없음", [], None, ["CFG lv=8 sim=8 tick=200 jit=2,4,6 (v2 cfg)", "CFG lv=8 sim=8 tick=100 jit=2,4,6"]),
):
    link, fake = link_with(lines)
    n_ping = len(fake.out)
    link.push_config(levels=8, max_simultaneous=8, tick_period_ms=100, jitter_ticks=[2, 4, 6])
    s = Sim([(10, bytes(fake.out[n_ping:]))], 50)
    got = [m for _, m in s.said("CFG")]
    check(link.fw_version == want_v and got == want_cfg,
          f"PicoLink 펌웨어 {label}: 판단 v{link.fw_version}, 펌웨어 응답 {got}")
check(T.parse_fw_version("PONG v=3 f=1") == 3 and T.parse_fw_version("PICO_READY v2 cells=24") == 2
      and T.parse_fw_version("CFG lv=8") is None, "펌웨어 버전 판독 (배너·PONG)")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("9. 칸 → PCA 채널 배열 (2026-10-05, 보드마다 세로 2칸씩)과 파도 시험 도구")
print("=" * 72)
from tactile_vision import geometry as geo  # noqa: E402

out = subprocess.run([str(EXE), "--map"], capture_output=True, text=True).stdout.split("\n")
fw_map = {int(a): (int(b), int(c)) for a, b, c in (ln.split() for ln in out if ln and not ln.startswith("MAP"))}
py_map = {c: (geo.PCA_ADDRS.index(a), ch) for c, (a, ch) in enumerate(geo.CELL_TO_PCA)}
check(fw_map == py_map, "펌웨어 CELL_TO_CHANNEL = Python geometry.CELL_TO_PCA (24칸)")
check(any(ln == "MAP colpairs" for ln in out), "펌웨어 배열 이름 colpairs (PING의 map= 값)")
want_rows = [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]]
ok = all([geo.pca_of(r * 6 + c)[1] for c in range(6)] == want_rows[r % 2]
         and {geo.pca_of(r * 6 + c)[0] for c in range(6)} == {geo.PCA_ADDRS[r // 2]} for r in range(4))
check(ok, "1·3행 채널 0 2 4 6 8 10, 2·4행 1 3 5 7 9 11 / 1~2행 0x40, 3~4행 0x41")
check(all(sorted(geo.pca_of(c)[1] for c in range(b * 12, b * 12 + 12)) == list(range(12)) for b in (0, 1)),
      "보드마다 채널 0~11이 한 번씩 (중복·누락 없음)")
check({geo.pca_of(r * 6 + c)[1] for r in range(4) for c in range(3)} == set(range(6)),
      "왼손 열 0~2 = 채널 0~5, 오른손 열 3~5 = 채널 6~11")

sys.path.insert(0, str(REPO / "tools"))
import servo_wave_test as W  # noqa: E402

order = [u[0] for u in W.units_for("channel")]
check(order == [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 12, 18, 13, 19, 14, 20, 15, 21, 16, 22, 17, 23],
      f"파도 '채널 순서' = 0x40 채널 0→11, 0x41 채널 0→11 (칸 {order[:6]} …)")
check([u[0] for u in W.units_for("cell")] == list(range(24))
      and [len(u) for u in W.units_for("column")] == [4] * 6, "파도 '칸 순서' 0→23, '열 순서' 6열 × 4칸")

# 파도 도구가 보낼 GRID(50 Hz)를 시뮬레이션 시각으로 만들어 실제 펌웨어 코드에 넣는다 (박자 40 ms, 16단계)
wave = W.Wave(W.units_for("channel"), gap=0.25, bump=1.6, loops=1)
LOWT, HIGHT = 232, 372
ev = [(10 + c, P.set_limit_frame(c, LOWT, HIGHT, c)) for c in range(N)]
ev.append((100, P.config_frame(16, 24, 40, (2, 4, 6), 30)))
t0 = 1200  # 실제 도구처럼 아래 위치에서 1 s 넘게 기다린 뒤 시작 (기동 기본값 372 → 시험 범위 아래 232로 이동한 다음)
for k in range(int(t0 / 20)):
    ev.append((200 + 20 * k, P.grid_frame(np.zeros(N, dtype=int), None, (40 + k) % 256)))
base = int(t0 / 20)
for k in range(int(wave.duration / 0.02) + 25):
    ev.append((t0 + 20 * k, P.grid_frame(wave.levels_at(k * 0.02, 15), None, (40 + base + k) % 256)))
s = Sim(ev, t0 + int(wave.duration * 1000) + 600)
peak = {}
for c in range(N):
    hi = [(t, v) for t, v in s.targets[c] if t >= t0]
    if hi:
        peak[c] = max(hi, key=lambda tv: (tv[1], -tv[0]))
check(len(peak) == N and all(v == HIGHT for _, v in peak.values()), f"[levels] 24칸 모두 위({HIGHT} tick)까지 올라감")
peaks = [peak[c][0] for c in order]
check(all(b > a for a, b in zip(peaks, peaks[1:])), "[levels] 핀이 올라가는 순서 = 계획한 채널 순서")
gaps = np.diff(peaks)
check(np.all(np.abs(gaps - 250) <= 45), f"[levels] 이웃 핀 사이 간격 {gaps.min()}~{gaps.max()} ms (계획 250 ms ± 박자 40 ms)")
end_ok = all(s.tick_at(c, t0 + int(wave.duration * 1000) + 500) == LOWT for c in range(N))
check(end_ok, f"[levels] 파도가 끝나면 모두 아래({LOWT} tick)로 돌아옴")
moving = max(sum(1 for c in range(N) if s.tick_at(c, t) != LOWT) for t in range(t0, t0 + int(wave.duration * 1000), 20))
planned = wave.max_raised()
check(moving <= planned + 2, f"[levels] 동시에 아래가 아닌 핀 최대 {moving}개 (계획 {planned}개 + 2 이내)")
steps = [abs(b - a) for c in range(N) for (_, a), (_, b) in zip(s.writes[c][1:], s.writes[c][2:])]
check(max(steps) <= 4, f"[levels] 슬루 제한 유지: 한 번에 최대 {max(steps)} tick")

# smooth 모드(기본): 도구와 같은 방식으로 RAW_TICK(바뀐 칸만, 50 Hz) + KEEPALIVE를 만들어 넣는다
sw = W.Wave(W.units_for("channel"), gap=W.DEFAULT_GAP["channel"], bump=4.0, loops=2)
raised = [int(np.count_nonzero(sw.frac_at(k * 0.02) > 1e-6)) for k in range(5, int(sw.duration / 0.02) - 5)]
check(min(raised) >= 1, f"[smooth] 끊김 없음: 첫 핀이 오른 뒤 끝날 때까지 늘 1개 이상 올라가 있음 (최소 {min(raised)}, 최대 {max(raised)})")
check(abs(sw.duration - (2 - 1) * sw.period - (23 * sw.gap + sw.bump)) < 1e-9 and sw.period == 12.0,
      f"[smooth] 기본 한 바퀴 {sw.period:g} s(간격 {sw.gap:g} s × 24), 2바퀴 {sw.duration:g} s — 바퀴 사이 쉼 없이 이어짐")
try:
    W.Wave(W.units_for("channel"), gap=0.1, bump=4.0)
    check(False, "[smooth] 오르내림이 한 바퀴보다 길면 거부")
except ValueError:
    check(True, "[smooth] 오르내림이 한 바퀴보다 길면 거부 (gap 0.1 × 24 = 2.4 s < bump 4 s)")
ev = [(10 + c, P.set_limit_frame(c, LOWT, HIGHT, c)) for c in range(N)]
ev.append((100, P.config_frame(8, 24, 200, (2, 4, 6), 30)))
ev += [(200 + 20 * k, P.grid_frame(np.zeros(N, dtype=int), None, (40 + k) % 256)) for k in range(50)]
t0, seq, sent, last_tx = 1300, 100, np.full(N, LOWT), 0
for k in range(int(sw.duration / 0.02) + 1):
    t_ms = t0 + 20 * k
    ticks = sw.ticks_at(k * 0.02, LOWT, HIGHT)
    for c in np.nonzero(ticks != sent)[0]:
        ev.append((t_ms, P.raw_tick_frame(int(c), int(ticks[c]), seq % 256)))
        seq += 1
        sent[c] = ticks[c]
        last_tx = t_ms
    if t_ms - last_tx > 200:
        ev.append((t_ms, P.keepalive_frame(seq % 256)))
        seq += 1
        last_tx = t_ms
end_ms = t0 + int(sw.duration * 1000)
s = Sim(ev, end_ms + 400)
check(not s.said("FAILSAFE") or all(t < t0 for t, _ in s.said("FAILSAFE")), "[smooth] 시험 중 failsafe 없음 (KEEPALIVE)")
w_after = {c: [(t, v) for t, v in s.writes[c] if t >= t0] for c in range(N)}
jump = max(abs(b - a) for c in range(N) for (_, a), (_, b) in zip(w_after[c], w_after[c][1:]))
max_step = math.ceil(140 * math.pi / 4.0 * 0.02)
check(jump <= max_step, f"[smooth] 연속 동작: 20 ms마다 최대 {jump} tick씩 움직임 (≈{jump / 2.1619:.2f}°, 계단 없음)")
first_peak = [min(t for t, v in w_after[c] if v == HIGHT) for c in order]
check(all(b > a for a, b in zip(first_peak, first_peak[1:])), "[smooth] 핀이 위에 닿는 순서 = 채널 순서")
g = np.diff(first_peak)
check(np.all(np.abs(g - 500) <= 25), f"[smooth] 이웃 핀 간격 {g.min()}~{g.max()} ms (계획 500 ms)")
n_up = {c: sum(1 for (_, a), (_, b) in zip(w_after[c], w_after[c][1:]) if b == HIGHT and a != HIGHT) for c in range(N)}
check(set(n_up.values()) == {2}, f"[smooth] 2바퀴 = 칸마다 위에 2번 ({sorted(set(n_up.values()))})")
check(all(s.tick_at(c, end_ms + 300) == LOWT for c in range(N)), f"[smooth] 끝나면 모두 아래({LOWT} tick)")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("10. 끊김·종료 시 천천히 맨 아래로 (2026-10-05)")
print("=" * 72)
s = Sim([], 50)
first = {c: s.writes[c][0] for c in range(N)}
check(all(v == (0, HOME) for v in first.values()),
      f"Pico가 켜지면 젯슨 연결 전에도 24칸 모두 이 장치의 맨 아래({HOME} tick ≈ {(HOME * 4.8828125 - 500) / 1900 * 180:.1f}°)로")
# 실제 장치 범위(사용자 캘리브레이션 372/232)로: 맨 위에서 종료(HOME_ALL)
ev = [(10 + c, P.set_limit_frame(c, HOME, MAX, c)) for c in range(N)]
ev.append((50, P.config_frame(8, 24, 200, (2, 4, 6), 35)))  # 젯슨이 기동 시 보내는 설정 (동시 구동 24)
ev += stream(100, 1000, 66, lambda k: grid({c: 7 for c in range(N)}, seq=40 + k))
ev.append((1100, P.home_all_frame(90)))
s = Sim(ev, 3500)
arr = [next((t for t, v in s.writes[c] if t > 1100 and v == HOME), None) for c in range(N)]
check(all(s.tick_at(c, 1099) == MAX for c in range(N)) and all(a is not None for a in arr),
      f"맨 위({MAX})에 있던 24칸 모두 종료 후 맨 아래({HOME})에 도착")
check(max(arr) - 1100 <= 1450 and min(arr) - 1100 >= 1350,
      f"범위 끝→끝 {abs(HOME - MAX)} tick을 {min(arr) - 1100}~{max(arr) - 1100} ms에 (≈1.4 s, 24칸 함께)")
check(all(max(abs(b - a) for (_, a), (_, b) in zip([w for w in s.writes[c] if w[0] >= 1100],
                                                  [w for w in s.writes[c] if w[0] >= 1100][1:])) == 1
          for c in range(N)), "내려오는 동안 모든 칸이 1 tick씩만 움직임")
# 내려오는 중에 새 값이 오면(다시 실행) 보통 속도로 그 값을 따른다
ev2 = ev + stream(1500, 2500, 66, lambda k: grid({0: 3}, seq=120 + k))
s = Sim(ev2, 2600)
check(s.tick_at(0, 2500) == LV[3], f"내려오는 중에 다시 실행하면 새 값(레벨 3 = {LV[3]} tick)으로 정상 동작")
# 내려오는 중에 RAW_TICK(캘리브레이션)이 오면 즉시 그 값
ev3 = ev + [(1300, P.raw_tick_frame(5, 300, 130))] + [(t, P.keepalive_frame(131 + i)) for i, t in enumerate(range(1400, 2600, 200))]
s = Sim(ev3, 2600)
check(s.tick_at(5, 1300) == 300 and s.tick_at(5, 2500) == 300, "내려오는 중 RAW_TICK은 즉시 적용되고 그대로 유지")
hdr = (REPO / "firmware/pico/tactile_pico/tactile_core.h").read_text(encoding="utf-8")
fw_def = {k: int(re.search(rf"constexpr uint16_t {k} = (\d+);", hdr).group(1)) for k in ("HOME_TICK_DEFAULT", "MAX_TICK_DEFAULT")}
from tactile_vision import calibration as CAL  # noqa: E402

check(fw_def == {"HOME_TICK_DEFAULT": CAL.DEVICE_HOME_TICK, "MAX_TICK_DEFAULT": CAL.DEVICE_MAX_TICK} == {
      "HOME_TICK_DEFAULT": HOME, "MAX_TICK_DEFAULT": MAX},
      f"펌웨어 기동 기본값 = Python 장치 기본 범위 = 이 시험 기준 (home {HOME}, max {MAX})")
dt = CAL.CalibTable.default()
check(all((c.home_tick, c.max_tick) == (HOME, MAX) for c in dt.channels),
      "캘리브레이션 파일이 없을 때 젯슨 기본값도 같은 범위·방향 (가까우면 핀 위)")
real = CAL.CalibTable.load(REPO / "calib/servo_calib.json")
tt = CAL.CalibTable.for_tool(REPO / "calib/servo_calib.json", 8, None, None, "t")
check(all((a == b).all() for a, b in zip(tt.ends(), real.ends())),
      f"시험 도구 범위 기본값 = 캘리브레이션 파일 ({tt.range_text()})")
t2 = CAL.CalibTable.for_tool(REPO / "calib/servo_calib.json", 8, 360, 100, "t")
d2, u2 = t2.ends()
check(all(d2 == 360) and all(u2 == (260 if real.channels[0].inverted else 460)),
      f"--home 360 --span 100 → 맨 아래 360, 맨 위 {u2[0]} (올라가는 방향은 파일을 따름)")

tmp.cleanup()
print()
print("FAIL 0 — 전부 통과" if not FAIL else f"FAIL {FAIL}건")
sys.exit(1 if FAIL else 0)
