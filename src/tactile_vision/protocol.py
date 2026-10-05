"""
Jetson ↔ Pico 시리얼 프로토콜.

기존 ASCII "GRID:v0,...,v23\\n" 방식의 문제:
  - 체크섬이 없어 값 하나가 깨지면 서보가 튄다. 사람 손에 닿는 장치에서 허용 불가.
  - 최대 100바이트 → USB CDC 버퍼 오버플로 시 패킷 경계가 깨진다.
  - 파싱에 String/strtok을 쓰면 RP2040 힙이 장시간 구동에서 단편화된다.

여기서는 고정 길이 바이너리 프레임 + CRC8을 쓴다.
셀 1바이트에 레벨(4bit) + 위험도(4bit)를 함께 실어 24바이트로 끝낸다.

프레임 구조 (Jetson → Pico)
  offset  크기  내용
  0       1    SYNC0 = 0xAA
  1       1    SYNC1 = 0x55
  2       1    TYPE
  3       1    SEQ   (0~255 순환, 유실 감지용)
  4       1    LEN   (payload 바이트 수)
  5       LEN  payload
  5+LEN   1    CRC8  (TYPE..payload 전체, poly 0x07, init 0x00)

Pico → Jetson 은 저빈도이므로 디버깅 편의를 위해 ASCII 한 줄을 쓴다.

v3 (2026-10-03, firmware/pico/tactile_pico/tactile_core.h)
  · 젯슨은 계산이 끝날 때마다 GRID를 보내고, Pico가 자기 박자(SET_CONFIG의 tick_period_ms,
    기본 200 ms)마다 최신 값 하나만 적용한다. 진동은 박자의 2·4번째 1/4 구간.
  · SET_CONFIG 7바이트: levels, max_simultaneous, tick_period_ms(u16), 위험도 1/2/3 진동 tick.
    4바이트 v2 형식(levels, max_simultaneous, v2 진동 반주기 u16)도 받는다.
  · KEEPALIVE(0x07): 내용 없음. failsafe(500 ms 무통신 → 전체 홈)만 막는다 — 캘리브레이션 도구 전용.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SYNC0 = 0xAA
SYNC1 = 0x55
HEADER_LEN = 5
CRC_LEN = 1

# ── 프레임 타입 ──
TYPE_GRID = 0x01  # 24셀 레벨+위험도
TYPE_RAW_TICK = 0x02  # 캘리브레이션: 채널에 tick 직접 인가
TYPE_HOME_ALL = 0x03  # 전체 홈 복귀
TYPE_SET_LIMIT = 0x04  # 채널 home/max tick 설정
TYPE_PING = 0x05  # 생존 확인
TYPE_SET_CONFIG = 0x06  # 런타임 파라미터
TYPE_KEEPALIVE = 0x07  # v3: failsafe 타이머만 갱신 (캘리브레이션 도구용)

# ── SET_CONFIG 범위 (펌웨어 tactile_core.h와 같아야 한다) ──
TICK_PERIOD_MS_MIN = 20
TICK_PERIOD_MS_MAX = 2000
JITTER_TICKS_MAX = 40
LEGACY_JITTER_HALF_MS = 75  # v2 펌웨어 기본 진동 반주기

# ── FLAGS (GRID payload 마지막 바이트) ──
FLAG_NONE = 0x00
FLAG_ESTOP = 0x01  # 즉시 전체 홈, 이후 GRID 무시

CELL_COUNT = 24
LEVEL_MASK = 0x0F
HAZARD_SHIFT = 4
MAX_LEVEL = 15
MAX_HAZARD = 15


def crc8(data: bytes | bytearray, poly: int = 0x07, init: int = 0x00) -> int:
    """CRC-8/ATM. Pico 쪽 구현과 반드시 동일해야 한다."""
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_frame(ftype: int, payload: bytes | bytearray, seq: int) -> bytes:
    if len(payload) > 255:
        raise ValueError(f"payload가 255바이트 초과: {len(payload)}")
    body = bytes([ftype, seq & 0xFF, len(payload)]) + bytes(payload)
    return bytes([SYNC0, SYNC1]) + body + bytes([crc8(body)])


def pack_cells(levels: np.ndarray, hazards: np.ndarray | None = None) -> bytes:
    """
    levels:  (24,) 0~15 정수 — 핀 높이 단계
    hazards: (24,) 0~15 정수 — jitter 강도 (0=없음)
    """
    lv = np.asarray(levels, dtype=np.int16).reshape(-1)
    if lv.size != CELL_COUNT:
        raise ValueError(f"셀 수 불일치: {lv.size} != {CELL_COUNT}")
    if np.any((lv < 0) | (lv > MAX_LEVEL)):
        raise ValueError(f"레벨 범위 초과: {lv.min()}~{lv.max()}")

    if hazards is None:
        hz = np.zeros(CELL_COUNT, dtype=np.int16)
    else:
        hz = np.asarray(hazards, dtype=np.int16).reshape(-1)
        if hz.size != CELL_COUNT:
            raise ValueError(f"위험도 수 불일치: {hz.size}")
        if np.any((hz < 0) | (hz > MAX_HAZARD)):
            raise ValueError(f"위험도 범위 초과: {hz.min()}~{hz.max()}")

    return bytes(((hz << HAZARD_SHIFT) | lv).astype(np.uint8))


def unpack_cells(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """pack_cells의 역변환. 테스트/디버그용."""
    arr = np.frombuffer(data[:CELL_COUNT], dtype=np.uint8)
    return (arr & LEVEL_MASK).astype(np.int16), (arr >> HAZARD_SHIFT).astype(np.int16)


def grid_frame(
    levels: np.ndarray,
    hazards: np.ndarray | None = None,
    seq: int = 0,
    flags: int = FLAG_NONE,
) -> bytes:
    payload = pack_cells(levels, hazards) + bytes([flags])
    return build_frame(TYPE_GRID, payload, seq)


def raw_tick_frame(channel: int, tick: int, seq: int = 0) -> bytes:
    """캘리브레이션용: 채널에 PWM tick을 직접 인가."""
    if not 0 <= channel < CELL_COUNT:
        raise ValueError(f"채널 범위 초과: {channel}")
    if not 0 <= tick < 4096:
        raise ValueError(f"tick 범위 초과: {tick}")
    return build_frame(TYPE_RAW_TICK, struct.pack(">BH", channel, tick), seq)


def home_all_frame(seq: int = 0) -> bytes:
    return build_frame(TYPE_HOME_ALL, b"", seq)


def set_limit_frame(channel: int, home_tick: int, max_tick: int, seq: int = 0) -> bytes:
    """채널별 캘리브레이션 결과를 Pico에 반영."""
    if not 0 <= channel < CELL_COUNT:
        raise ValueError(f"채널 범위 초과: {channel}")
    for t in (home_tick, max_tick):
        if not 0 <= t < 4096:
            raise ValueError(f"tick 범위 초과: {t}")
    return build_frame(TYPE_SET_LIMIT, struct.pack(">BHH", channel, home_tick, max_tick), seq)


def ping_frame(seq: int = 0) -> bytes:
    return build_frame(TYPE_PING, b"", seq)


def keepalive_frame(seq: int = 0) -> bytes:
    return build_frame(TYPE_KEEPALIVE, b"", seq)


def _check_common_config(levels: int, max_simultaneous: int) -> None:
    if not 2 <= levels <= 16:
        raise ValueError(f"levels 범위 초과(2~16): {levels}")
    if not 1 <= max_simultaneous <= CELL_COUNT:
        raise ValueError(f"max_simultaneous 범위 초과(1~{CELL_COUNT}): {max_simultaneous}")


def config_frame(
    levels: int,
    max_simultaneous: int,
    tick_period_ms: int,
    jitter_ticks: tuple[int, int, int] | list[int],
    seq: int = 0,
) -> bytes:
    """
    v3 런타임 파라미터 (7바이트). 펌웨어 재플래시 없이 실험 조건을 바꾼다.
      tick_period_ms  핀 갱신 박자 (20~2000 ms, 기본 200 = 초당 5회)
      jitter_ticks    위험도 1/2/3의 진동 진폭 (tick, 0~40)
    범위 밖이면 펌웨어는 그 항목을 무시하므로 여기서 미리 거부한다.
    """
    _check_common_config(levels, max_simultaneous)
    if not TICK_PERIOD_MS_MIN <= tick_period_ms <= TICK_PERIOD_MS_MAX:
        raise ValueError(f"tick_period_ms 범위 초과({TICK_PERIOD_MS_MIN}~{TICK_PERIOD_MS_MAX}): "
                         f"{tick_period_ms}")
    jt = [int(v) for v in jitter_ticks]
    if len(jt) != 3 or any(not 0 <= v <= JITTER_TICKS_MAX for v in jt):
        raise ValueError(f"jitter_ticks는 0~{JITTER_TICKS_MAX} 정수 3개: {jitter_ticks}")
    return build_frame(
        TYPE_SET_CONFIG,
        struct.pack(">BBHBBB", levels, max_simultaneous, tick_period_ms, *jt),
        seq,
    )


def config_frame_legacy(
    levels: int, max_simultaneous: int, jitter_half_ms: int = LEGACY_JITTER_HALF_MS, seq: int = 0
) -> bytes:
    """v2 펌웨어용 4바이트 형식. v3 펌웨어도 받지만 박자·진동 진폭은 바꾸지 않는다."""
    _check_common_config(levels, max_simultaneous)
    if not 20 <= jitter_half_ms <= 1000:
        raise ValueError(f"jitter_half_ms 범위 초과(20~1000): {jitter_half_ms}")
    return build_frame(
        TYPE_SET_CONFIG,
        struct.pack(">BBH", levels, max_simultaneous, jitter_half_ms),
        seq,
    )


# ─────────────────────────────────────────────────────────────────
# 수신측 파서 (테스트 및 Pico 시뮬레이터용)
# ─────────────────────────────────────────────────────────────────
@dataclass
class ParsedFrame:
    ftype: int
    seq: int
    payload: bytes


class FrameParser:
    """
    스트리밍 파서. 동기 바이트를 찾아 프레임을 복원하고 CRC를 검증한다.
    Pico 펌웨어의 상태 기계와 같은 논리를 따른다.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.stat_ok = 0
        self.stat_crc_err = 0
        self.stat_resync = 0

    def feed(self, data: bytes) -> list[ParsedFrame]:
        self.buf.extend(data)
        out: list[ParsedFrame] = []
        while True:
            # 동기 바이트 탐색
            i = self.buf.find(bytes([SYNC0, SYNC1]))
            if i < 0:
                # 마지막 1바이트는 SYNC0일 수 있으니 남긴다
                if len(self.buf) > 1:
                    self.buf = self.buf[-1:]
                break
            if i > 0:
                self.stat_resync += 1
                del self.buf[:i]
            if len(self.buf) < HEADER_LEN:
                break
            length = self.buf[4]
            total = HEADER_LEN + length + CRC_LEN
            if len(self.buf) < total:
                break
            body = bytes(self.buf[2 : HEADER_LEN + length])
            got = self.buf[HEADER_LEN + length]
            if crc8(body) == got:
                out.append(ParsedFrame(body[0], body[1], body[3:]))
                self.stat_ok += 1
                del self.buf[:total]
            else:
                self.stat_crc_err += 1
                del self.buf[:2]  # 동기 바이트를 버리고 재탐색
        return out


def frame_size(payload_len: int) -> int:
    return HEADER_LEN + payload_len + CRC_LEN


GRID_FRAME_BYTES = frame_size(CELL_COUNT + 1)  # 31 = 헤더 5 + 셀 24 + FLAGS 1 + CRC 1
