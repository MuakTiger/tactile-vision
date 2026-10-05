"""
실내/실외 자동 전환 정책.

  SceneSwitcher  프레임마다 (돌릴 깊이 모델, d_far)를 정한다.

동작 (scene.mode = auto)
  1. 장면 판별(Places365)은 초당 rate_hz회만 돌린다. 장면은 프레임 단위로 바뀌지 않는다.
  2. P(실외) → SceneGate(시간 EMA tau_s + 히스테리시스 + 체류 dwell_s) → 확정 장면.
  3. 돌릴 모델: 확정 장면의 모델 하나. 애매하면(uncertain_policy = both) 두 모델을
     모두 돌리고, 파이프라인이 셀마다 더 가까운 값을 쓴다 — 판정이 틀려도 장애물을
     멀게 알리는 쪽으로는 틀리지 않게. 애매함 = 전환 대기 중이거나 평활 P(실외)가
     두 문턱 사이(SceneGate.uncertain). hold면 확정 장면의 모델 하나만 돌린다.
  4. d_far: 확정 장면의 값(기본 실내 4 m / 실외 8 m). 애매 구간에도 바뀌지 않는다.
     장면이 바뀌면 ln(d_far)를 crossfade_s 동안 선형 보간한다(v2 부록 C-3).
     보간 중에 또 바뀌면 지금 값에서 새 목표로 다시 보간한다(값이 튀지 않음).

scene.mode = indoor / outdoor 는 판별 없이 그 모델과 d_far로 고정한다(실험 기준 조건).

시각(now)은 호출자가 준다. 파이프라인은 프레임 캡처 시각(perf_counter)을 쓴다.
녹화 영상 입력이면 영상 시간이 아니라 처리 시간 기준이다(매핑 필터와 같은 기준).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import SceneCfg
from .scene import SceneGate, SceneResult

logger = logging.getLogger(__name__)

SCENES = ("indoor", "outdoor")
KO = {"indoor": "실내", "outdoor": "실외", "both": "실내+실외"}


class Classifier(Protocol):
    def classify(self, bgr: np.ndarray) -> SceneResult: ...


@dataclass(frozen=True)
class SceneDecision:
    """한 프레임의 장면 결정. 파이프라인 결과·CSV·화면 표시에 그대로 쓴다."""

    state: str  # 확정 장면 indoor | outdoor
    models: tuple[str, ...]  # 이번 프레임에 돌릴 깊이 모델 (1개 또는 2개)
    far_m: float  # 이번 프레임 d_far (크로스페이드 반영)
    uncertain: bool  # 애매함(전환 대기 중 또는 두 문턱 사이). 고정 모드는 항상 False
    p_out: float | None  # 가장 최근 판별의 P(실외). 고정 모드는 None
    p_smooth: float | None  # 평활 P(실외)
    classified: bool  # 이번 프레임에 판별기를 돌렸는지
    clf_ms: float  # 이번 프레임의 판별 시간 (안 돌렸으면 0)
    top1: str | None  # 가장 최근 판별의 1위 장면 이름

    @property
    def models_label(self) -> str:
        return "both" if len(self.models) > 1 else self.models[0]


class SceneSwitcher:
    """
    switcher = SceneSwitcher(cfg.scene, SceneClassifier(...))
    d = switcher.step(frame.image, now=frame.ts)
    → d.models 로 깊이 모델을 돌리고, mapper.set_far(d.far_m)
    """

    def __init__(self, cfg: SceneCfg, classifier: Classifier | None = None) -> None:
        if cfg.mode not in ("auto", *SCENES):
            raise ValueError(f"SceneSwitcher는 auto | indoor | outdoor 만: {cfg.mode}")
        if cfg.mode == "auto" and classifier is None:
            raise ValueError("scene.mode=auto에는 장면 판별기가 필요함")
        self.cfg = cfg
        self.mode = cfg.mode
        self.classifier = classifier if cfg.mode == "auto" else None
        self.gate: SceneGate | None = None
        if self.mode == "auto":
            self.gate = SceneGate(tau_s=cfg.tau_s, enter_outdoor=cfg.enter_outdoor,
                                  enter_indoor=cfg.enter_indoor, dwell_s=cfg.dwell_s)
        self.period = 1.0 / cfg.rate_hz
        self.far_of = {"indoor": cfg.far_indoor_m, "outdoor": cfg.far_outdoor_m}
        self.n_classified = 0

        self._next_clf: float | None = None  # 다음 판별 예정 시각
        self._last: SceneResult | None = None
        # d_far 크로스페이드: _from → _to, _t0부터 crossfade_s 동안. _from None = 보간 끝
        self._far_from: float | None = None
        self._far_to: float | None = None
        self._far_t0 = 0.0

    # ── 상태 ──
    @property
    def state(self) -> str | None:
        """확정 장면. auto는 첫 판별 전까지 None."""
        return self.gate.state if self.gate is not None else self.mode

    @property
    def n_switches(self) -> int:
        return self.gate.n_switches if self.gate is not None else 0

    # ── d_far 크로스페이드 ──
    def far_at(self, now: float) -> float:
        """지금의 d_far. ln(d_far)를 시간에 선형 보간한다."""
        if self._far_to is None:
            raise RuntimeError("장면이 아직 정해지지 않음 — step()을 먼저 호출")
        T = self.cfg.crossfade_s
        if self._far_from is None or T <= 0:
            return self._far_to
        a = (now - self._far_t0) / T
        if a >= 1.0:
            self._far_from = None
            return self._far_to
        if a <= 0.0:
            return self._far_from  # 전환 순간 = 시작값 그대로 (exp(ln x)의 반올림 오차 없이)
        return math.exp((1.0 - a) * math.log(self._far_from) + a * math.log(self._far_to))

    def _retarget_far(self, far: float, now: float) -> None:
        if self._far_to is None:  # 첫 판정은 보간 없이 바로
            self._far_to = far
            return
        if far == self._far_to:
            return
        cur = self.far_at(now)  # 보간 중이면 지금 값에서 다시 시작
        self._far_from, self._far_to, self._far_t0 = cur, far, now

    # ── 판별 주기 ──
    def _due(self, now: float) -> bool:
        """초당 rate_hz회. 예정 시각을 누적해 프레임률과 무관하게 평균 횟수를 맞춘다."""
        if self._next_clf is None or now >= self._next_clf:
            nxt = (now if self._next_clf is None else self._next_clf) + self.period
            # 프레임이 끊겼다 오면 밀린 판별을 몰아서 하지 않는다
            self._next_clf = nxt if nxt > now else now + self.period
            return True
        return False

    # ── 메인 ──
    def step(self, bgr: np.ndarray, now: float | None = None) -> SceneDecision:
        now = time.perf_counter() if now is None else now
        classified, ms = False, 0.0
        if self.classifier is not None and self._due(now):
            assert self.gate is not None
            first = self.gate.state is None
            res = self.classifier.classify(bgr)
            self.gate.observe(res.p_outdoor, now=now)
            self._last = res
            self.n_classified += 1
            classified, ms = True, res.ms
            if first:
                logger.info("장면 첫 판정: %s (P(실외) %.2f, 1위 %s)", KO[self.gate.state],
                            res.p_outdoor, res.top[0][0] if res.top else "-")

        state = self.state
        if state is None:  # auto인데 판별기가 한 번도 안 돈 경우 — _due가 첫 호출에 True라 오지 않음
            raise RuntimeError("장면 판정 없음")
        self._retarget_far(self.far_of[state], now)
        far = self.far_at(now)

        uncertain = self.gate is not None and self.gate.uncertain
        both = uncertain and self.cfg.uncertain_policy == "both"
        last = self._last
        return SceneDecision(
            state=state,
            models=SCENES if both else (state,),
            far_m=far,
            uncertain=uncertain,
            p_out=None if last is None else float(last.p_outdoor),
            p_smooth=None if self.gate is None else self.gate.p,
            classified=classified,
            clf_ms=ms,
            top1=None if last is None or not last.top else last.top[0][0],
        )

    def describe(self) -> str:
        c = self.cfg
        if self.mode != "auto":
            return f"장면 고정: {KO[self.mode]} 모델, d_far {self.far_of[self.mode]:.1f}m (판별 안 함)"
        return (
            f"장면 자동 전환: Places365 초당 {c.rate_hz:g}회, 평활 {c.tau_s:g}s, "
            f"실외 > {c.enter_outdoor:.2f} / 실내 < {c.enter_indoor:.2f}, 체류 {c.dwell_s:g}s, "
            f"애매하면 {'두 모델 중 가까운 값' if c.uncertain_policy == 'both' else '확정 장면 모델 유지'} | "
            f"d_far 실내 {c.far_indoor_m:.1f}m / 실외 {c.far_outdoor_m:.1f}m, "
            f"전환 시 {c.crossfade_s:g}s 로그 보간"
        )
