"""
셀 거리(미터) → 핀 레벨(정수) 변환.

설계 근거
─────────
스트로크 20.9mm를 손바닥 높이 식별 한계(~2.5mm)로 나누면 8단계가 상한이다.
8단계밖에 없으므로 "어느 거리대에 단계를 쓸 것인가"가 설계의 전부다.

  linear   : 거리에 균등. far_m 밖이 전부 포화 → 원거리 정보 소실.
  log      : Weber-Fechner. 근거리 촘촘 + 원거리 유지. 기본값.
  disparity: 모델 native(1/d). 근거리에 과도하게 몰림.
  relative_ema: 프레임별 min-max 금지. 수 초 시간상수로 분위수를 추적해
               스케일을 안정화한 상대 모드. 비교 실험용.

프레임별 min-max 정규화는 의도적으로 제공하지 않는다. 같은 핀 높이가
장면마다 다른 거리를 뜻하게 되어 사용자가 매핑을 학습할 수 없고,
깊이 분산이 작은 장면에서는 모델 노이즈를 손바닥으로 증폭 전달한다.
"""

from __future__ import annotations

import math

import numpy as np

from .config import FilterCfg, MappingCfg


def level_edges(cfg: MappingCfg, far_m: float | None = None) -> np.ndarray:
    """단계 경계 거리 배열 (길이 levels+1, 가까운 쪽부터). 문서/시각화용."""
    near = cfg.near_m
    far = cfg.far_m if far_m is None else far_m
    n = cfg.levels
    if cfg.mode == "linear":
        return np.linspace(near, far, n + 1)
    if cfg.mode == "log":
        return np.geomspace(near, far, n + 1)
    if cfg.mode == "disparity":
        inv = np.linspace(1.0 / near, 1.0 / far, n + 1)
        return 1.0 / inv
    # relative_ema는 런타임 분위수에 의존 → 정적 경계 없음
    return np.geomspace(near, far, n + 1)


class DepthMapper:
    """
    상태를 가지는 매핑기. 프레임마다 update()를 호출한다.

    파이프라인: 거리 → EMA 평활 → 연속 레벨 → 히스테리시스 → 확정프레임 → 정수 레벨
    """

    def __init__(self, cfg: MappingCfg, fcfg: FilterCfg, n_cells: int) -> None:
        self.cfg = cfg
        self.fcfg = fcfg
        self.n = n_cells
        self.levels = cfg.levels
        self.top = cfg.levels - 1

        # 상태
        self._dist_ema: np.ndarray | None = None
        self._level = np.zeros(n_cells, dtype=np.int16)
        self._pending = np.zeros(n_cells, dtype=np.int16)
        self._pending_cnt = np.zeros(n_cells, dtype=np.int16)
        # relative_ema 모드용 분위수 추적
        self._q_near: float | None = None
        self._q_far: float | None = None
        # 외부에서 정한 d_far (장면별 d_far·크로스페이드). None = cfg.far_m
        self._far_override: float | None = None
        self._active_far = cfg.far_m

    # ── d_far 지정 ──
    def set_far(self, far_m: float | None) -> None:
        """
        d_far를 바꾼다. 실내/실외 전환이 프레임마다 호출한다(update 전에).
        None이면 설정값 mapping.far_m으로 돌아간다. 필터 상태(거리 EMA)는 유지한다.
        """
        if far_m is not None and not far_m > self.cfg.near_m:
            raise ValueError(f"d_far {far_m}가 near {self.cfg.near_m} 이하")
        self._far_override = None if far_m is None else float(far_m)
        if not self.cfg.adaptive_far:
            self._active_far = self._base_far()

    def _base_far(self) -> float:
        return self.cfg.far_m if self._far_override is None else self._far_override

    # ── 상태 초기화 ──
    def reset(self) -> None:
        self._dist_ema = None
        self._level[:] = 0
        self._pending[:] = 0
        self._pending_cnt[:] = 0
        self._q_near = self._q_far = None
        self._active_far = self._base_far()

    # ── 연속 레벨 산출 ──
    def _continuous(self, d: np.ndarray) -> np.ndarray:
        """거리(m) → 연속 레벨 [0, levels-1]. 값이 클수록 가까움."""
        c = self.cfg
        near = c.near_m
        far = self._active_far
        d = np.clip(d, near, far)

        if c.mode == "linear":
            frac = (far - d) / (far - near)
        elif c.mode == "log":
            frac = (math.log(far) - np.log(d)) / (math.log(far) - math.log(near))
        elif c.mode == "disparity":
            frac = (1.0 / d - 1.0 / far) / (1.0 / near - 1.0 / far)
        elif c.mode == "relative_ema":
            lo = self._q_near if self._q_near is not None else near
            hi = self._q_far if self._q_far is not None else far
            hi = max(hi, lo + 0.1)
            frac = (hi - np.clip(d, lo, hi)) / (hi - lo)
        else:  # config.validate에서 이미 걸러짐
            raise AssertionError(c.mode)

        return np.clip(frac, 0.0, 1.0) * self.top

    # ── 적응형 far bound ──
    def _update_far(self, d_raw: np.ndarray) -> None:
        c = self.cfg
        base = self._base_far()
        if not c.adaptive_far:
            self._active_far = base
            return
        # 모든 셀이 d_far 밖이면 = 경로 클리어 → 원거리 분해능 확보를 위해 확장
        nearest = float(np.min(d_raw))
        if nearest > base:
            target = min(c.adaptive_far_max_m, nearest * 1.6)
        else:
            target = base
        # 급변 방지: 느린 추종
        self._active_far += 0.1 * (target - self._active_far)

    # ── relative_ema 분위수 추적 ──
    def _update_quantiles(self, d_raw: np.ndarray, dt: float) -> None:
        if self.cfg.mode != "relative_ema":
            return
        lo = float(np.percentile(d_raw, 5))
        hi = float(np.percentile(d_raw, 95))
        tau = max(1e-3, self.cfg.ema_window_s)
        a = 1.0 - math.exp(-max(1e-3, dt) / tau)
        if self._q_near is None:
            self._q_near, self._q_far = lo, hi
        else:
            self._q_near += a * (lo - self._q_near)
            self._q_far += a * (hi - self._q_far)

    # ── 메인 ──
    def update(self, dist_m: np.ndarray, dt: float = 1 / 15) -> np.ndarray:
        """
        dist_m: 셀별 대표 거리 (미터), shape (n_cells,)
        반환:   셀별 정수 레벨 0..levels-1 (클수록 가까움 = 핀 높음)

        필터는 방향 비대칭이다. 접근(위험)은 빠르게, 후퇴(안전)는 느리게.
        """
        d_raw = np.asarray(dist_m, dtype=np.float32).reshape(-1)
        if d_raw.size != self.n:
            raise ValueError(f"셀 수 불일치: {d_raw.size} != {self.n}")
        # 모델이 0/음수/NaN을 낼 수 있음 → d_far 밖으로 처리 (없는 것으로 간주)
        fill = max(self.cfg.far_m, self._base_far())
        d_raw = np.where(np.isfinite(d_raw) & (d_raw > 1e-3), d_raw, fill)

        self._update_far(d_raw)
        self._update_quantiles(d_raw, dt)

        # 1. 비대칭 EMA — 거리가 줄어들면(접근) 빠르게 따라가고,
        #    늘어나면(후퇴) 천천히 놓아준다.
        f = self.fcfg
        if self._dist_ema is None:
            self._dist_ema = d_raw.copy()
        else:
            approaching = d_raw < self._dist_ema
            alpha = np.where(approaching, f.ema_alpha_approach, f.ema_alpha_recede)
            self._dist_ema += alpha * (d_raw - self._dist_ema)

        # 2. 연속 레벨
        x = self._continuous(self._dist_ema)

        # 3. 비대칭 히스테리시스
        #    상승(접근)은 좁은 데드밴드, 하강(후퇴)은 넓은 데드밴드.
        prev = self._level.astype(np.float32)
        rising = x > prev
        band = np.where(rising, 0.5 + f.hysteresis_up, 0.5 + f.hysteresis_down)
        moved = np.abs(x - prev) > band
        # 한 프레임에 여러 단계를 건너뛰지 않도록 하강은 1단계씩 추종.
        # 접근 시엔 즉시 목표까지 올리는 것이 안전하므로 상승은 제한 없음.
        # 주의: np.clip(want, prev-1, want)는 prev-1 > want일 때 numpy가
        # 상한을 우선 적용해 제한이 무력화된다. maximum을 써야 한다.
        want = np.floor(x + 0.5)
        step_limited = np.maximum(want, prev - 1.0)
        target = np.where(moved, np.where(rising, want, step_limited), prev)
        target = np.clip(target, 0, self.top).astype(np.int16)

        # 4. 방향별 확정 프레임
        need = np.where(
            target > self._level, f.confirm_frames_up, f.confirm_frames_down
        ).astype(np.int16)
        need = np.maximum(need, 1)
        same_pending = target == self._pending
        self._pending_cnt = np.where(same_pending, self._pending_cnt + 1, 1)
        self._pending = target
        commit = (self._pending_cnt >= need) & (target != self._level)
        self._level = np.where(commit, target, self._level).astype(np.int16)

        return self._level.copy()

    # ── 진단 ──
    @property
    def active_far_m(self) -> float:
        return self._active_far

    @property
    def smoothed_dist(self) -> np.ndarray | None:
        return None if self._dist_ema is None else self._dist_ema.copy()

    def describe_edges(self) -> str:
        e = level_edges(self.cfg, self._active_far)
        prof = self.cfg.profile
        rows = [
            f"매핑 {self.cfg.mode}  {self.cfg.near_m}~{self._active_far:.1f}m  "
            f"{self.levels}단계  단계당 {prof.step_mm:.2f}mm"
        ]
        for lv in range(self.top, -1, -1):
            lo, hi = e[self.top - lv], e[self.top - lv + 1]
            rows.append(
                f"  레벨 {lv}  {lo:5.2f}~{hi:5.2f}m  폭 {hi-lo:4.2f}m  "
                f"핀 {prof.level_to_mm(lv):5.1f}mm"
            )
        return "\n".join(rows)
