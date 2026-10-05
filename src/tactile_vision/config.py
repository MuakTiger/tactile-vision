"""설정 로딩 + 검증. 물리/지각 제약 위반은 기동 시점에 실패시킨다."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import geometry as geo

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CameraCfg:
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    flip_horizontal: bool = False
    queue_size: int = 1
    # 자동초점: False = 끄고 focus로 고정, True = 켬, None = 카메라 설정을 건드리지 않음
    autofocus: bool | None = None
    # 자동초점을 끌 때 고정할 초점값 (None = 렌즈 위치 그대로)
    focus: int | None = None


@dataclass
class DepthCfg:
    engine: str = "models/da2_metric_hypersim_vits_504x280.engine"  # 실내 모델
    # 실외 모델. scene.mode가 auto / outdoor일 때 로드한다
    outdoor_engine: str = "models/da2_metric_vkitti_vits_504x280.engine"
    input_spec: str = "wide"
    max_depth_m: float = 20.0
    metric: bool = True

    @property
    def spec(self) -> geo.InputSpec:
        if self.input_spec not in geo.INPUT_SPECS:
            raise ValueError(
                f"input_spec '{self.input_spec}' 없음. 가능: {list(geo.INPUT_SPECS)}"
            )
        return geo.INPUT_SPECS[self.input_spec]


@dataclass
class GridCfg:
    reducer: str = "percentile"
    percentile: float = 10.0
    cell_margin: float = 0.0


@dataclass
class MappingCfg:
    mode: str = "log"
    near_m: float = 0.5
    far_m: float = 8.0
    levels: int = 8
    stroke_mm: float = 20.9
    ema_window_s: float = 3.0
    adaptive_far: bool = False
    adaptive_far_max_m: float = 15.0

    @property
    def profile(self) -> geo.StrokeProfile:
        return geo.StrokeProfile(stroke_mm=self.stroke_mm, levels=self.levels)


@dataclass
class FilterCfg:
    """
    비대칭 필터. 오검출 비용이 방향에 따라 다르다는 안전 원칙을 반영한다.

      접근(거리 감소 / 레벨 상승) = 위험 신호 → 빠르고 민감하게
      후퇴(거리 증가 / 레벨 하강) = 안전 신호 → 느리고 둔감하게

    핀이 늦게 올라가는 것(미탐지)은 충돌로 이어지지만,
    핀이 늦게 내려가는 것(잔존)은 안전하다.
    """

    ema_alpha_approach: float = 0.65  # 거리 감소 시 EMA 계수 (크면 빠름)
    ema_alpha_recede: float = 0.25  # 거리 증가 시 EMA 계수
    hysteresis_up: float = 0.15  # 레벨 상승 데드밴드 (단계 비율)
    hysteresis_down: float = 0.45  # 레벨 하강 데드밴드
    confirm_frames_up: int = 1  # 상승 확정 프레임
    confirm_frames_down: int = 3  # 하강 확정 프레임


@dataclass
class HazardCfg:
    enabled: bool = True
    engine: str = "models/yolov8n.engine"
    conf: float = 0.40
    danger_levels: dict[str, int] = field(default_factory=dict)
    max_alert_m: float = 4.0
    max_jitter_cells: int = 6
    # 위험도 1/2/3의 진동 진폭(도). 펌웨어 v3에 tick으로 보낸다(SG90 명목 1도 = 2.162 tick).
    jitter_deg: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])


@dataclass
class SceneCfg:
    """
    실내/실외 자동 전환. Places365 장면 판별 → 깊이 모델과 d_far를 고른다.

      auto     자동 전환 (장치 기본)
      indoor   실내 모델 + far_indoor_m 고정 (판별 안 함, 실험 기준 조건)
      outdoor  실외 모델 + far_outdoor_m 고정
      off      예전 방식: depth.engine 하나 + mapping.far_m

    기본값 off: scene 섹션이 없는 설정 파일은 예전처럼 동작한다.
    """

    mode: str = "off"
    checkpoint_dir: str = "checkpoints/places365"
    rate_hz: float = 5.0  # 장면 판별 횟수/초
    tau_s: float = 1.0  # P(실외) 시간 평활 시상수
    enter_outdoor: float = 0.65  # 평활 P(실외)가 이 값을 넘은 채 dwell_s 유지 → 실외
    enter_indoor: float = 0.35  # 이 값 아래로 dwell_s 유지 → 실내
    dwell_s: float = 1.5
    uncertain_policy: str = "both"  # both | hold
    far_indoor_m: float = 4.0
    far_outdoor_m: float = 8.0
    crossfade_s: float = 0.7  # 전환 시 ln(d_far) 선형 보간 시간


@dataclass
class TransportCfg:
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    # 핀 갱신 박자(Hz). Pico 펌웨어 v3가 이 박자마다 최신 값 하나를 적용하고, 그 사이에
    # 박자의 2배 빈도로 위험도 진동을 넣는다. 젯슨은 계산 결과가 나올 때마다 보낸다.
    pin_rate_hz: float = 5.0
    # 젯슨 → Pico 전송 상한(Hz). Pico가 박자에 맞춰 최신 값만 쓰므로 과전송 방지용일 뿐. 0 = 제한 없음
    max_rate_hz: float = 30.0
    failsafe_ms: int = 500


@dataclass
class SafetyCfg:
    max_simultaneous_moves: int = 8
    startup_home_delay_ms: int = 800


@dataclass
class CalibrationCfg:
    file: str = "calib/servo_calib.json"


@dataclass
class RuntimeCfg:
    log_level: str = "INFO"
    metrics_csv: str = "logs/metrics.csv"


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    grid: GridCfg = field(default_factory=GridCfg)
    mapping: MappingCfg = field(default_factory=MappingCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    hazard: HazardCfg = field(default_factory=HazardCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    transport: TransportCfg = field(default_factory=TransportCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    calibration: CalibrationCfg = field(default_factory=CalibrationCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)

    # ── 경로 해석 ──
    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    # ── 검증 ──
    def validate(self) -> list[str]:
        """치명적 문제는 예외, 경고는 리스트로 반환."""
        warns: list[str] = []
        m, f = self.mapping, self.filter

        c = self.camera
        if c.autofocus not in (None, True, False):
            raise ValueError(f"camera.autofocus는 true/false/null: {c.autofocus}")
        if c.focus is not None and (not isinstance(c.focus, int) or c.focus < 0):
            raise ValueError(f"camera.focus는 0 이상 정수 또는 null: {c.focus}")
        if c.autofocus is True and c.focus is not None:
            warns.append("camera.autofocus가 true라 camera.focus는 무시됨")

        s = self.scene
        if s.mode not in {"auto", "indoor", "outdoor", "off"}:
            raise ValueError(f"scene.mode는 auto | indoor | outdoor | off: {s.mode}")
        if s.rate_hz <= 0:
            raise ValueError(f"scene.rate_hz는 0보다 커야 함: {s.rate_hz}")
        if not 0.0 < s.enter_indoor < s.enter_outdoor < 1.0:
            raise ValueError(
                f"scene 문턱은 0 < enter_indoor({s.enter_indoor}) < "
                f"enter_outdoor({s.enter_outdoor}) < 1 이어야 함"
            )
        if s.tau_s < 0 or s.dwell_s < 0 or s.crossfade_s < 0:
            raise ValueError("scene.tau_s, dwell_s, crossfade_s는 0 이상")
        if s.uncertain_policy not in {"both", "hold"}:
            raise ValueError(f"scene.uncertain_policy는 both | hold: {s.uncertain_policy}")
        if min(s.far_indoor_m, s.far_outdoor_m) <= m.near_m:
            raise ValueError(
                f"scene.far_indoor_m({s.far_indoor_m}), far_outdoor_m({s.far_outdoor_m})는 "
                f"mapping.near_m({m.near_m})보다 커야 함"
            )

        if m.near_m <= 0 or m.far_m <= m.near_m:
            raise ValueError(f"거리 범위 오류: near={m.near_m} far={m.far_m}")
        if m.levels < 2:
            raise ValueError(f"levels는 2 이상이어야 함: {m.levels}")
        if m.mode not in {"linear", "log", "disparity", "relative_ema"}:
            raise ValueError(f"mapping.mode 미지원: {m.mode}")
        if self.grid.reducer not in {"percentile", "mean", "min"}:
            raise ValueError(f"grid.reducer 미지원: {self.grid.reducer}")
        if not 0.0 <= self.grid.cell_margin < 0.5:
            raise ValueError(f"cell_margin 범위 초과: {self.grid.cell_margin}")
        if not 0.0 < f.ema_alpha_approach <= 1.0:
            raise ValueError(f"ema_alpha_approach 범위 초과: {f.ema_alpha_approach}")
        if not 0.0 < f.ema_alpha_recede <= 1.0:
            raise ValueError(f"ema_alpha_recede 범위 초과: {f.ema_alpha_recede}")
        if f.ema_alpha_approach < f.ema_alpha_recede:
            warns.append(
                "ema_alpha_approach < ema_alpha_recede — 접근보다 후퇴에 "
                "빠르게 반응하는 설정. 안전 원칙에 반함"
            )
        if f.hysteresis_up > f.hysteresis_down:
            warns.append(
                "hysteresis_up > hysteresis_down — 접근 검출이 후퇴보다 "
                "둔감함. 안전 원칙에 반함"
            )
        if f.confirm_frames_up > f.confirm_frames_down:
            warns.append("confirm_frames_up > confirm_frames_down — 안전 원칙에 반함")

        # 스트로크가 물리 한계를 넘는지
        max_stroke = geo.SG90_RANGE_DEG * geo.MM_PER_DEG
        if m.stroke_mm > max_stroke:
            raise ValueError(
                f"stroke_mm {m.stroke_mm}가 SG90 물리 한계 {max_stroke:.1f}mm 초과"
            )
        if m.stroke_mm > geo.RACK_LENGTH_MM:
            raise ValueError(f"stroke_mm이 랙 길이 {geo.RACK_LENGTH_MM}mm 초과")

        # 지각 한계 검증 — 이게 이 프로젝트의 핵심 제약
        prof = m.profile
        if not prof.perceptually_valid:
            warns.append(
                f"단계 간격 {prof.step_mm:.2f}mm < 손바닥 식별 한계 "
                f"{geo.PALM_HEIGHT_JND_MM}mm — levels를 "
                f"{geo.max_useful_levels(m.stroke_mm)} 이하로 줄일 것"
            )

        # 핀 갱신 박자 (Pico 펌웨어 v3) — 서보 속도·진동 시간 대비
        t = self.transport
        if not 0.5 <= t.pin_rate_hz <= 50.0:
            raise ValueError(f"transport.pin_rate_hz 범위 초과(0.5~50): {t.pin_rate_hz}")
        if t.max_rate_hz < 0:
            raise ValueError(f"transport.max_rate_hz는 0 이상(0 = 제한 없음): {t.max_rate_hz}")
        pin_period_ms = 1000.0 / t.pin_rate_hz
        step_time_ms = geo.travel_time_s(prof.step_mm) * 1000
        if pin_period_ms < step_time_ms:
            warns.append(
                f"핀 갱신 주기 {pin_period_ms:.0f}ms < 1단계 이동시간 {step_time_ms:.0f}ms — "
                f"서보가 다음 갱신 전에 도착하지 못함. pin_rate_hz를 {1000/step_time_ms:.0f} 이하로"
            )
        # 진동은 박자의 2·4번째 1/4 구간. 서보 PWM 한 주기(20 ms)보다 짧으면 진동이 불규칙해진다.
        quarter_ms = pin_period_ms / 4
        pwm_period_ms = 1000.0 / geo.PWM_FREQ_HZ
        if quarter_ms < pwm_period_ms:
            warns.append(
                f"진동 한 구간 {quarter_ms:.1f}ms < 서보 PWM 주기 {pwm_period_ms:.0f}ms — "
                "진동이 불규칙할 수 있음(pin_rate_hz를 낮출 것)"
            )
        if t.max_rate_hz and t.max_rate_hz < t.pin_rate_hz:
            warns.append(
                f"전송 상한 max_rate_hz({t.max_rate_hz}) < pin_rate_hz({t.pin_rate_hz}) — "
                "박자마다 새 값이 오지 않는다"
            )

        hz = self.hazard
        if len(hz.jitter_deg) != 3 or any(d < 0 for d in hz.jitter_deg):
            raise ValueError(f"hazard.jitter_deg는 0 이상 값 3개(위험도 1/2/3): {hz.jitter_deg}")
        jt = [geo.deg_delta_to_ticks(d) for d in hz.jitter_deg]
        if max(jt) > 40:
            raise ValueError(f"hazard.jitter_deg 최대 {40 / geo.TICKS_PER_DEG:.1f}도(40 tick) 초과: "
                             f"{hz.jitter_deg}")
        small = [f"{d:g}도={k}tick({k * geo.TICK_US:.1f}µs)" for d, k in zip(hz.jitter_deg, jt)
                 if 0 < d and k * geo.TICK_US <= geo.SG90_DEADBAND_US]
        if small:
            warns.append(
                f"진동 진폭 {', '.join(small)} ≤ SG90 데드밴드 약 {geo.SG90_DEADBAND_US:.0f}µs"
                "(데이터시트 명목) — 실제로 안 움직일 수 있음. 실기에서 확인할 것"
            )

        # 전류 추정
        est_a = self.safety.max_simultaneous_moves * 0.25
        if self.safety.max_simultaneous_moves > geo.CELL_COUNT:
            raise ValueError("max_simultaneous_moves가 셀 수 초과")
        warns.append(
            f"동시 이동 {self.safety.max_simultaneous_moves}개 → 추정 피크 "
            f"{est_a:.1f}A @5V ({est_a*5:.0f}W). 벅 정격 확인 필요"
        )

        if self.hazard.max_jitter_cells > self.safety.max_simultaneous_moves:
            warns.append(
                f"max_jitter_cells({self.hazard.max_jitter_cells}) > "
                f"max_simultaneous_moves({self.safety.max_simultaneous_moves}) — "
                "jitter가 전류 상한을 넘을 수 있음"
            )

        # 종횡비 정합성
        spec = self.depth.spec
        if abs(spec.aspect - geo.ARRAY_ASPECT) > 0.01:
            ratio = spec.aspect / geo.ARRAY_ASPECT
            warns.append(
                f"입력 종횡비 {spec.aspect:.2f} ≠ 배열 종횡비 "
                f"{geo.ARRAY_ASPECT:.2f} — 셀이 수평 방향으로 {ratio:.2f}배 "
                "넓은 각도를 담당(비등방 매핑)"
            )
        return warns


def _merge(dc: Any, data: dict) -> Any:
    """dict를 dataclass에 주입. 미지의 키는 에러로 잡는다(오타 방지)."""
    known = {fld.name for fld in dc.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{type(dc).__name__}에 알 수 없는 키: {sorted(unknown)}")
    for k, v in data.items():
        setattr(dc, k, v)
    return dc


def load(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        path = REPO_ROOT / "config" / "default.yaml"
    path = Path(path)
    if not path.exists():
        logger.warning("설정 파일 없음(%s) — 기본값 사용", path)
        return cfg

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in cfg.__dataclass_fields__:
        if section in raw:
            _merge(getattr(cfg, section), raw[section] or {})
    unknown = set(raw) - set(cfg.__dataclass_fields__)
    if unknown:
        raise ValueError(f"설정 최상위에 알 수 없는 섹션: {sorted(unknown)}")
    return cfg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = load()
    print(geo.summary())
    print("\n── 설정 검증 ──")
    for w in c.validate():
        print(f"  [주의] {w}")
    print(f"\n매핑: {c.mapping.mode} {c.mapping.near_m}~{c.mapping.far_m}m")
    print(f"프로파일: {c.mapping.profile.describe()}")
    print(f"입력: {c.depth.spec.describe()}")
