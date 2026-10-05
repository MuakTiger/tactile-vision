"""
하드웨어 기하 상수 — 전부 STL 실측에서 유도됨.

측정 출처: hardware/full/7_22 set_final.stl, parts/{rack,pinion,servo_bracket}.stl
측정 방법: 바이너리 STL 파싱 → union-find 바디 분리 → 바운딩박스 / FFT 잇수 검출
측정 일자: 2026-09-22

이 파일의 숫자를 바꾸려면 반드시 재측정 근거를 같이 남길 것.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────
# 1. 촉각 배열 기하 (full.stl 실측)
# ─────────────────────────────────────────────────────────────────
#   베이스판          173.00 × 200.00 × 34.75 mm
#   서보유닛 24개     27.00 ×  20.00 × 27.30 mm
#   배치              200mm축에 6개(피치 30.0mm), 173mm축에 4개(피치 40.0mm)
#   전체 높이         62.30 mm
GRID_COLS = 6
GRID_ROWS = 4
CELL_COUNT = GRID_COLS * GRID_ROWS  # 24

# 칸 → PCA9685 (I2C 주소, 채널) — 2026-10-05 배선. 칸 번호는 화면 순서(행 우선)이고,
# 1~2행은 0x40 보드, 3~4행은 0x41 보드. 보드마다 왼쪽 위부터 세로 2칸이 한 세트:
#   위 줄 채널 0 2 4 6 8 10 / 아래 줄 1 3 5 7 9 11  (채널 0~5 = 왼손 열 0~2, 6~11 = 오른손 열 3~5)
# 펌웨어 firmware/pico/tactile_pico/tactile_core.h CELL_TO_CHANNEL과 같아야 한다 (tools/test_firmware.py가 확인).
PCA_ADDRS = (0x40, 0x41)


def pca_of(cell: int) -> tuple[int, int]:
    """칸 번호 → (PCA9685 I2C 주소, 채널)."""
    if not 0 <= cell < CELL_COUNT:
        raise ValueError(f"칸 번호 범위 초과: {cell}")
    r, c = divmod(cell, GRID_COLS)
    return PCA_ADDRS[r // 2], c * 2 + (r % 2)


CELL_TO_PCA = tuple(pca_of(i) for i in range(CELL_COUNT))

COL_PITCH_MM = 30.0  # 가로(200mm축) 서보 중심간 거리
ROW_PITCH_MM = 40.0  # 세로(173mm축) 서보 중심간 거리

ARRAY_SPAN_W_MM = COL_PITCH_MM * (GRID_COLS - 1)  # 150.0 — 중심~중심
ARRAY_SPAN_H_MM = ROW_PITCH_MM * (GRID_ROWS - 1)  # 120.0
BASE_PLATE_W_MM = 200.0
BASE_PLATE_H_MM = 173.0

# 배열 종횡비 — 시야 크롭 종횡비를 여기에 맞추면 공간 대응이 왜곡되지 않음
ARRAY_ASPECT = ARRAY_SPAN_W_MM / ARRAY_SPAN_H_MM  # 1.25

# ─────────────────────────────────────────────────────────────────
# 2. 랙 & 피니언 (parts 실측)
# ─────────────────────────────────────────────────────────────────
#   rack    55.86 × 10.05 × 3.17 mm  (36개 인쇄: 24 사용 + 12 여분)
#   pinion  외경 18.68mm, 잇수 12 (FFT 주기 검출: 강도 950 vs 2차 454)
#   bracket 24.68 × 26.23 × 4.46 mm
#
#   검증: OD = module × (z + 2) = 1.3333 × 14 = 18.667mm  vs  실측 18.68mm → 일치
#   따라서 설계 의도는 피치원 직경 16.00mm 정수값.
PINION_TEETH = 12
PINION_MODULE_MM = 4.0 / 3.0  # 1.3333
PINION_PITCH_DIA_MM = PINION_MODULE_MM * PINION_TEETH  # 16.00
PINION_OUTER_DIA_MM = PINION_MODULE_MM * (PINION_TEETH + 2)  # 18.667

RACK_LENGTH_MM = 55.86  # 랙 전체 길이 — 스트로크 상한이 아니라 서보 각도가 제한

# 핵심 변환 상수: 피니언 1회전당 랙 직선 이동량
MM_PER_REV = math.pi * PINION_PITCH_DIA_MM  # 50.265
MM_PER_DEG = MM_PER_REV / 360.0  # 0.139626
DEG_PER_MM = 1.0 / MM_PER_DEG  # 7.1620

# ─────────────────────────────────────────────────────────────────
# 3. SG90 서보 / PCA9685 PWM
# ─────────────────────────────────────────────────────────────────
PWM_FREQ_HZ = 50
PWM_TICKS = 4096
TICK_US = 1_000_000.0 / PWM_FREQ_HZ / PWM_TICKS  # 4.8828 µs

SG90_MIN_US = 500.0  # 명목 0°
SG90_MAX_US = 2400.0  # 명목 180°
SG90_RANGE_DEG = 180.0

# SG90 무부하 속도: 0.1 s / 60° @ 4.8V → 600 °/s. 부하 고려 안전계수 0.6 적용.
SG90_SPEED_DEG_PER_S = 600.0 * 0.6  # 360 °/s
SG90_SPEED_MM_PER_S = SG90_SPEED_DEG_PER_S * MM_PER_DEG  # 50.3 mm/s

# 서보 각도를 전부 쓰지 않는 이유: 기구 간섭 + 스톨 회피 여유.
# 실제 가용 범위는 채널별 캘리브레이션이 덮어씀 (calib/servo_calib.json).
USABLE_DEG_DEFAULT = 150.0
STROKE_MM_DEFAULT = USABLE_DEG_DEFAULT * MM_PER_DEG  # 20.94


def us_to_tick(us: float) -> int:
    """마이크로초 → PCA9685 tick (50Hz, 4096분할)."""
    return int(round(us / TICK_US))


def tick_to_us(tick: int) -> float:
    return tick * TICK_US


def deg_to_us(deg: float) -> float:
    """SG90 명목 선형 매핑. 채널 캘리브레이션이 있으면 그걸 쓸 것."""
    frac = max(0.0, min(1.0, deg / SG90_RANGE_DEG))
    return SG90_MIN_US + frac * (SG90_MAX_US - SG90_MIN_US)


def deg_to_tick(deg: float) -> int:
    return us_to_tick(deg_to_us(deg))


# 각도 변화 → tick 변화 (진동 진폭용). 명목 1도 = 1900/180 us = 10.56 us = 2.162 tick.
TICKS_PER_DEG = (SG90_MAX_US - SG90_MIN_US) / SG90_RANGE_DEG / TICK_US
# SG90 데드밴드: 데이터시트 명목 약 10 us(≈ 2 tick). 이보다 작은 변화는 서보가 무시할 수 있다.
# 실물 미검증 — 1도 진동이 실제로 움직이는지 실기에서 확인할 것.
SG90_DEADBAND_US = 10.0


def deg_delta_to_ticks(deg: float) -> int:
    """각도 변화량(도) → tick 변화량. 위험도 진동 진폭을 펌웨어에 보낼 때 쓴다."""
    return int(round(abs(deg) * TICKS_PER_DEG))


def mm_to_deg(mm: float) -> float:
    return mm * DEG_PER_MM


def travel_time_s(delta_mm: float) -> float:
    """핀이 delta_mm 이동하는 데 걸리는 시간 (부하 보정 속도 기준)."""
    return abs(delta_mm) / SG90_SPEED_MM_PER_S


# ─────────────────────────────────────────────────────────────────
# 4. 지각 한계 기반 단계 수 산출
# ─────────────────────────────────────────────────────────────────
# 손바닥 높이차 식별 한계는 약 2~3mm (2점 식별 한계 10~11mm와는 별개 축).
# 보수적으로 2.5mm를 하한으로 두고 단계 수를 정한다.
PALM_HEIGHT_JND_MM = 2.5


def max_useful_levels(stroke_mm: float = STROKE_MM_DEFAULT) -> int:
    """
    주어진 스트로크에서 손바닥이 실제로 구분 가능한 최대 단계 수.

    레벨 0을 0mm, 레벨 L-1을 stroke_mm로 두면 간격은 L-1개다.
    간격 >= JND 조건: stroke/(L-1) >= JND  →  L <= stroke/JND + 1
    """
    return max(2, int(stroke_mm / PALM_HEIGHT_JND_MM) + 1)


@dataclass(frozen=True)
class StrokeProfile:
    """스트로크와 단계 수를 묶어 검증하는 값 객체."""

    stroke_mm: float
    levels: int

    @property
    def step_mm(self) -> float:
        """인접 단계 사이의 핀 높이차. 간격 수는 levels-1개다."""
        return self.stroke_mm / (self.levels - 1)

    @property
    def usable_deg(self) -> float:
        return mm_to_deg(self.stroke_mm)

    @property
    def perceptually_valid(self) -> bool:
        """단계 간격이 손바닥 식별 한계 이상인가."""
        return self.step_mm >= PALM_HEIGHT_JND_MM

    def level_to_mm(self, level: int) -> float:
        """레벨 → 핀 높이(mm). 레벨 levels-1이 정확히 stroke_mm이 된다."""
        return level * self.step_mm

    def level_to_deg(self, level: int) -> float:
        return mm_to_deg(self.level_to_mm(level))

    def describe(self) -> str:
        ok = "OK" if self.perceptually_valid else "단계 간격이 지각 한계 미달"
        return (
            f"스트로크 {self.stroke_mm:.1f}mm / {self.levels}단계 = "
            f"간격 {self.step_mm:.2f}mm (각도 {self.usable_deg:.0f}°, "
            f"레벨0=0mm → 레벨{self.levels-1}={self.level_to_mm(self.levels-1):.1f}mm) [{ok}]"
        )


DEFAULT_PROFILE = StrokeProfile(stroke_mm=STROKE_MM_DEFAULT, levels=8)


# ─────────────────────────────────────────────────────────────────
# 5. 모델 입력 해상도 후보
# ─────────────────────────────────────────────────────────────────
# 제약 3개를 동시에 만족해야 함:
#   (a) DINOv2 패치 14의 배수
#   (b) 폭이 GRID_COLS(6)의 배수  — 셀 균등 분할
#   (c) 높이가 GRID_ROWS(4)의 배수
# → 폭 ∈ lcm(6,14)=42의 배수, 높이 ∈ lcm(4,14)=28의 배수
@dataclass(frozen=True)
class InputSpec:
    name: str
    width: int
    height: int

    def __post_init__(self) -> None:
        assert self.width % 14 == 0 and self.height % 14 == 0, "패치 14 배수 위반"
        assert self.width % GRID_COLS == 0, "폭이 열 수로 나눠지지 않음"
        assert self.height % GRID_ROWS == 0, "높이가 행 수로 나눠지지 않음"

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def cell_w(self) -> int:
        return self.width // GRID_COLS

    @property
    def cell_h(self) -> int:
        return self.height // GRID_ROWS

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def describe(self) -> str:
        return (
            f"{self.name}: {self.width}×{self.height} (종횡비 {self.aspect:.2f}) "
            f"셀 {self.cell_w}×{self.cell_h}px, {self.pixels:,}px"
        )


# 배열 종횡비(1.25)와 정확히 일치 — 공간 대응 왜곡 없음. FOV는 좁아짐.
INPUT_FAITHFUL = InputSpec("faithful", 420, 336)
# 카메라 16:9에 가까움(1.80) — FOV 넓음. 셀의 각도 종횡비가 비등방.
INPUT_WIDE = InputSpec("wide", 504, 280)
# wide와 같은 화각, 공식 입력(짧은 변 518)에 가장 가까운 크기(그리드 6×4 조건상 518 불가).
# 입력이 작으면 metric 모델이 거리를 멀게 낸다: 2026-10-03 같은 사진 4장, 셀 하위 10% 거리를
# 518 출력과 비교 — wide 실내 1.28·실외 1.21배, 924×504 실내 1.03·실외 1.00배.
# 연산은 wide의 약 4~6배(토큰 720 → 2376).
INPUT_WIDE_HI = InputSpec("wide_hi", 924, 504)

INPUT_SPECS = {s.name: s for s in (INPUT_FAITHFUL, INPUT_WIDE, INPUT_WIDE_HI)}


def summary() -> str:
    lines = [
        "── 하드웨어 실측 요약 ──",
        f"그리드            {GRID_COLS}×{GRID_ROWS} = {CELL_COUNT}셀",
        f"셀 피치           가로 {COL_PITCH_MM}mm / 세로 {ROW_PITCH_MM}mm",
        f"배열 스팬         {ARRAY_SPAN_W_MM}×{ARRAY_SPAN_H_MM}mm (종횡비 {ARRAY_ASPECT:.2f})",
        f"피니언            {PINION_TEETH}톱니, 모듈 {PINION_MODULE_MM:.4f}, "
        f"피치원 {PINION_PITCH_DIA_MM:.2f}mm",
        f"변환              {MM_PER_DEG:.6f} mm/° ({MM_PER_REV:.3f} mm/회전)",
        f"랙 길이           {RACK_LENGTH_MM}mm",
        f"PWM               {PWM_FREQ_HZ}Hz, 1tick = {TICK_US:.4f}µs, "
        f"{SG90_MIN_US:.0f}~{SG90_MAX_US:.0f}µs = tick {us_to_tick(SG90_MIN_US)}~{us_to_tick(SG90_MAX_US)}",
        f"기본 스트로크     {DEFAULT_PROFILE.describe()}",
        f"이론 최대 단계    {max_useful_levels()}단계 (JND {PALM_HEIGHT_JND_MM}mm 기준)",
        f"핀 최대이동 시간  {travel_time_s(STROKE_MM_DEFAULT)*1000:.0f}ms (전체 스트로크)",
        "",
        "모델 입력 후보:",
        f"  {INPUT_FAITHFUL.describe()}",
        f"  {INPUT_WIDE.describe()}",
        f"  {INPUT_WIDE_HI.describe()}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
