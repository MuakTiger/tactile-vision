"""프로토콜 검증: 왕복 무손실, CRC 오류 검출, 바이트 유실 후 재동기."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vision import protocol as P  # noqa: E402

FAIL = 0


def check(cond: bool, msg: str) -> None:
    global FAIL
    if cond:
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


print("=" * 70)
print("1. 프레임 크기 및 왕복")
print("=" * 70)
rng = np.random.default_rng(7)
levels = rng.integers(0, 8, 24)
hazards = rng.integers(0, 4, 24)
f = P.grid_frame(levels, hazards, seq=42)
print(f"  GRID 프레임 {len(f)}바이트: {f[:8].hex(' ')} ... {f[-4:].hex(' ')}")
check(len(f) == P.GRID_FRAME_BYTES == 31, f"GRID 프레임 31바이트 (실측 {len(f)})")
print(f"    = 헤더 {P.HEADER_LEN} + 셀 {P.CELL_COUNT} + FLAGS 1 + CRC {P.CRC_LEN}")

# 115200bps 기준 전송시간
tx_ms = len(f) * 10 / 115200 * 1000
print(f"  115200bps 전송시간 {tx_ms:.2f}ms (ASCII 100바이트 방식은 8.7ms)")
check(tx_ms < 3.0, f"전송시간 3ms 미만 (실측 {tx_ms:.2f}ms)")

p = P.FrameParser()
got = p.feed(f)
check(len(got) == 1, "프레임 1개 파싱")
check(got[0].ftype == P.TYPE_GRID and got[0].seq == 42, "타입/시퀀스 일치")
lv2, hz2 = P.unpack_cells(got[0].payload)
check(np.array_equal(lv2, levels), "레벨 24개 무손실 왕복")
check(np.array_equal(hz2, hazards), "위험도 24개 무손실 왕복")
check(got[0].payload[24] == P.FLAG_NONE, "FLAGS 바이트 위치 정확")

print("\n" + "=" * 70)
print("2. CRC 오류 검출 — 모든 바이트의 모든 비트를 뒤집어 검사")
print("=" * 70)
undetected = []
for byte_i in range(2, len(f)):  # 동기 바이트 제외
    for bit in range(8):
        bad = bytearray(f)
        bad[byte_i] ^= 1 << bit
        pp = P.FrameParser()
        res = pp.feed(bytes(bad))
        # 프레임이 통과했고 내용이 원본과 다르면 미검출
        if res:
            lv3, hz3 = P.unpack_cells(res[0].payload)
            if not (np.array_equal(lv3, levels) and np.array_equal(hz3, hazards)):
                undetected.append((byte_i, bit))
total_flips = (len(f) - 2) * 8
print(f"  단일비트 오류 {total_flips}건 중 미검출 {len(undetected)}건")
check(len(undetected) == 0, "모든 단일비트 오류를 검출 또는 무해화")

# 2비트 오류
rng2 = np.random.default_rng(11)
undet2 = 0
TRIALS = 3000
for _ in range(TRIALS):
    bad = bytearray(f)
    i, j = rng2.integers(2, len(f), 2)
    bi, bj = rng2.integers(0, 8, 2)
    bad[i] ^= 1 << bi
    bad[j] ^= 1 << bj
    if bytes(bad) == f:
        continue
    pp = P.FrameParser()
    res = pp.feed(bytes(bad))
    if res:
        lv3, hz3 = P.unpack_cells(res[0].payload)
        if not (np.array_equal(lv3, levels) and np.array_equal(hz3, hazards)):
            undet2 += 1
rate = undet2 / TRIALS
print(f"  2비트 오류 {TRIALS}건 중 미검출 {undet2}건 ({rate*100:.2f}%)")
check(rate < 0.01, f"2비트 오류 미검출률 1% 미만 (실측 {rate*100:.2f}%)")

print("\n" + "=" * 70)
print("3. 바이트 유실 후 재동기")
print("=" * 70)
stream = P.grid_frame(levels, hazards, 1) + P.grid_frame(levels, hazards, 2)
# 첫 프레임 중간에서 3바이트 유실
broken = stream[:10] + stream[13:]
pp = P.FrameParser()
res = pp.feed(broken)
print(f"  유실 스트림에서 복원된 프레임 {len(res)}개 "
      f"(정상 ok={pp.stat_ok} crc_err={pp.stat_crc_err} resync={pp.stat_resync})")
check(len(res) >= 1, "유실 후에도 다음 프레임 복원")
check(res[-1].seq == 2, "복원된 프레임이 두 번째(seq=2)")

print("\n" + "=" * 70)
print("4. 분할 수신 (USB CDC 조각화 모사)")
print("=" * 70)
pp = P.FrameParser()
collected = []
for k in range(0, len(stream), 7):  # 7바이트씩 도착
    collected += pp.feed(stream[k : k + 7])
check(len(collected) == 2, f"조각 수신에도 2프레임 복원 (실측 {len(collected)})")
check([c.seq for c in collected] == [1, 2], "순서 유지")

print("\n" + "=" * 70)
print("5. 기타 프레임 타입")
print("=" * 70)
for name, fr, exp_type in [
    ("RAW_TICK", P.raw_tick_frame(13, 350, 5), P.TYPE_RAW_TICK),
    ("HOME_ALL", P.home_all_frame(6), P.TYPE_HOME_ALL),
    ("SET_LIMIT", P.set_limit_frame(7, 120, 480, 8), P.TYPE_SET_LIMIT),
    ("PING", P.ping_frame(9), P.TYPE_PING),
    ("SET_CONFIG", P.config_frame(8, 8, 200, (2, 4, 6), 10), P.TYPE_SET_CONFIG),
    ("CONFIG_v2", P.config_frame_legacy(8, 8, 75, 11), P.TYPE_SET_CONFIG),
    ("KEEPALIVE", P.keepalive_frame(12), P.TYPE_KEEPALIVE),
]:
    pp = P.FrameParser()
    r = pp.feed(fr)
    ok = len(r) == 1 and r[0].ftype == exp_type
    print(f"  {name:<11} {len(fr):2d}바이트  파싱 {'OK' if ok else 'FAIL'}")
    check(ok, f"{name} 프레임 왕복")
cfg_pl = P.FrameParser().feed(P.config_frame(8, 6, 250, (3, 5, 7)))[0].payload
check(cfg_pl == bytes([8, 6, 0x00, 0xFA, 3, 5, 7]),
      f"SET_CONFIG v3 7바이트 = levels, max_sim, tick_period_ms(빅엔디언), 진동 tick×3 ({cfg_pl.hex(' ')})")
check(len(P.FrameParser().feed(P.config_frame_legacy(8, 8))[0].payload) == 4, "SET_CONFIG v2 4바이트")

print("\n" + "=" * 70)
print("6. 범위 검증 (잘못된 입력을 거부하는가)")
print("=" * 70)
for desc, fn in [
    ("레벨 16 초과", lambda: P.pack_cells(np.full(24, 16))),
    ("레벨 음수", lambda: P.pack_cells(np.full(24, -1))),
    ("셀 수 부족", lambda: P.pack_cells(np.zeros(23))),
    ("위험도 초과", lambda: P.pack_cells(np.zeros(24), np.full(24, 16))),
    ("채널 초과", lambda: P.raw_tick_frame(24, 300)),
    ("tick 초과", lambda: P.raw_tick_frame(0, 4096)),
    ("박자 19 ms (최소 20)", lambda: P.config_frame(8, 8, 19, (2, 4, 6))),
    ("박자 2001 ms (최대 2000)", lambda: P.config_frame(8, 8, 2001, (2, 4, 6))),
    ("진동 41 tick (최대 40)", lambda: P.config_frame(8, 8, 200, (2, 4, 41))),
    ("진동 값 2개", lambda: P.config_frame(8, 8, 200, (2, 4))),
    ("levels 17", lambda: P.config_frame(17, 8, 200, (2, 4, 6))),
    ("동시 구동 0", lambda: P.config_frame(8, 0, 200, (2, 4, 6))),
]:
    try:
        fn()
        check(False, f"{desc} 거부")
    except ValueError:
        check(True, f"{desc} 거부")

print("\n" + "=" * 70)
print(f"결과: {'전체 통과' if FAIL == 0 else f'{FAIL}개 실패'}")
print("=" * 70)
sys.exit(1 if FAIL else 0)
