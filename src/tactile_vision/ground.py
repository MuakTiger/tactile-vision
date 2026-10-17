"""
실외 땅 기준 거리 환산 — 상대깊이(단위 없음)를 카메라 높이·기울기·초점거리로 미터로 바꾼다.

원리
────
카메라가 땅에서 h 높이에 있고 아래로 θ 기울어 있으며 초점거리가 f이면, 화면 y줄에 보이는
땅까지의 깊이(광축 방향 거리) Z(y)를 계산할 수 있다. θ = 0이면 Z(y) = f·h / (y − cy).
상대깊이 모델의 출력 r은 역깊이에 비례한다고 본다: r = A / Z (하늘이 r ≈ 0이므로 이동 항은 0).
화면 아래 땅 줄들(Z ≤ band_max_m)에서 줄마다 r 중앙값 × Z(y)를 구하고, 그 중앙값을 A로 쓴다.
그러면 모든 픽셀의 거리 = A / r.

C920 1280×720, h 0.70 m, 수평이면 장치 입력(1280×698 → 924×504)의 맨 아래 줄이 약 1.82 m다.
그보다 가까운 땅은 화면에 없다.

근거 (2026-10-17 거리 사진, 논문 저장소 thesis/analysis/ground_scale.py)
  실외 60장(산책로·주차장, 1~8 m): MAPE 21.1% — 실내 모델 − 0.44 m 38.0%, 실외 metric 모델 519%.
  상자 거리로 맞춘 값 없음(h, f만 사용). 실내 60장은 54.8%로 나빠 실내에는 쓰지 않는다.
한계
  기울기 1°가 틀리면 실외 MAPE 약 ±5%p, 높이 5 cm면 약 ±5%p(같은 분석).
  땅이 화면 아래에 보여야 한다(사람이 많은 곳·계단·경사로는 가정이 깨진다).
  걸을 때 흔들림은 A를 로그 영역에서 시간 평활해 줄이지만, 이 부분은 검증하지 않았다.
"""

from __future__ import annotations

import logging
import math
import time

import torch

from .config import GroundCfg

logger = logging.getLogger(__name__)


def ground_depth_rows(height_px: int, f_px: float, cy_px: float, h_m: float, pitch_deg: float
                      ) -> tuple[list[int], list[float]]:
    """
    모델 입력 줄마다 그 줄에 보이는 땅의 깊이(광축 방향, m). 수평선 위 줄은 뺀다.
    f_px, cy_px는 모델 입력 좌표. pitch_deg > 0 = 아래를 봄. 반환: (줄 번호, 깊이)
    """
    th = math.radians(pitch_deg)
    rows, z = [], []
    for y in range(height_px):
        a = th + math.atan2(y - cy_px, f_px)  # 수평 아래로 잰 시선 각
        if a <= math.radians(0.5):
            continue
        d = h_m / math.tan(a)  # 땅 위 수평 거리
        rows.append(y)
        z.append(d * math.cos(th) + h_m * math.sin(th))  # 광축 방향 성분 = 모델이 내는 깊이
    return rows, z


class GroundScaler:
    """
    상대 역깊이 맵 (H, W) → 배율 A (r = A / Z). update()가 프레임마다 A를 갱신·평활한다.

    crop: 카메라 원본에서 장치 입력으로 자른 영역 (x, y, w, h) — camera.crop_rect와 같은 값.
    cfg_src_w: gcfg.f_px를 잰 원본 폭(camera.width). 실제 원본 폭이 다르면 f를 비례로 맞춘다.
    """

    def __init__(self, gcfg: GroundCfg, out_w: int, out_h: int,
                 crop: tuple[int, int, int, int], cfg_src_w: int, device: str = "cuda") -> None:
        self.gcfg = gcfg
        x, y, w, h = crop
        s = out_h / h  # 원본 px → 모델 px (세로)
        src_w = 2 * x + w  # 가운데 크롭
        src_h = 2 * y + h
        f_src = gcfg.f_px * src_w / cfg_src_w
        cy_src = src_h / 2 if gcfg.cy_px is None else gcfg.cy_px * src_w / cfg_src_w
        self.f = f_src * s
        self.cy = (cy_src - y) * s
        rows, z = ground_depth_rows(out_h, self.f, self.cy, gcfg.height_m, gcfg.pitch_deg)
        keep = [i for i, zz in enumerate(z) if zz <= gcfg.band_max_m]
        if len(keep) < gcfg.min_rows:
            raise ValueError(
                f"땅 줄이 {len(keep)}개뿐(최소 {gcfg.min_rows}) — 카메라 높이 {gcfg.height_m} m, "
                f"기울기 {gcfg.pitch_deg}°, band_max_m {gcfg.band_max_m} m로는 화면 아래에 땅이 거의 없다"
            )
        self.rows = torch.tensor([rows[i] for i in keep], dtype=torch.long, device=device)
        self.z = torch.tensor([z[i] for i in keep], dtype=torch.float32, device=device)
        self.z_near = min(z[i] for i in keep)
        self.z_far = max(z[i] for i in keep)
        self.raw = float("nan")  # 이번 프레임 A (평활 전)
        self.value = float("nan")  # 평활한 A
        self.n_invalid = 0
        self._ln: float | None = None
        self._t: float | None = None
        logger.info(
            "땅 기준 환산: 높이 %.2f m, 기울기 %+.1f°, f %.0f px(모델 %.1f), 땅 줄 %d개(%.2f~%.2f m), 평활 τ %.1f s",
            gcfg.height_m, gcfg.pitch_deg, f_src, self.f, len(keep), self.z_near, self.z_far,
            gcfg.smooth_tau_s,
        )

    def reset(self) -> None:
        self._ln = None
        self._t = None
        self.raw = self.value = float("nan")

    def estimate(self, rel: torch.Tensor) -> float:
        """한 프레임의 A (평활 없음). 줄마다 중앙값 → (중앙값 × Z)의 중앙값."""
        band = rel.index_select(0, self.rows)  # (n, W)
        med = torch.quantile(band.float(), 0.5, dim=1)
        return float(torch.quantile(med * self.z, 0.5))

    def update(self, rel: torch.Tensor, now: float | None = None) -> float:
        """A를 갱신하고 평활값을 반환. 무효(0 이하·NaN)면 직전 값을 유지한다(처음이면 NaN)."""
        now = time.perf_counter() if now is None else now
        a = self.estimate(rel)
        self.raw = a
        if not (math.isfinite(a) and a > 0):
            self.n_invalid += 1
            return self.value
        ln = math.log(a)
        tau = self.gcfg.smooth_tau_s
        if self._ln is None or tau <= 0 or self._t is None or now - self._t > 5 * tau:
            self._ln = ln  # 처음, 평활 끔, 또는 오래 쉬었다가(장면 전환 직후) 다시 쓸 때
        else:
            k = 1.0 - math.exp(-max(0.0, now - self._t) / tau)
            self._ln += k * (ln - self._ln)
        self._t = now
        self.value = math.exp(self._ln)
        return self.value
