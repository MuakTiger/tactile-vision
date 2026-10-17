"""
땅 기준 거리 환산·실내 보정 검증 (CPU만, 엔진·카메라·핀 안 씀).

  1. 땅 줄 깊이 기하 (수평 / 기울기)
  2. 합성 상대깊이 맵에서 배율 A 복원 (가림·잡음·무효)
  3. 시간 평활 (로그 EMA, 오래 쉰 뒤 초기화)
  4. 엔진 출력 → 미터 변환 (offset / ground)
  5. 설정 검증과 기본 설정

  python tools/test_ground.py
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision.camera import crop_rect  # noqa: E402
from tactile_vision.config import Config, GroundCfg, load  # noqa: E402
from tactile_vision.depth import DepthEngine  # noqa: E402
from tactile_vision.ground import GroundScaler, ground_depth_rows  # noqa: E402

FAIL = 0
W, H = 924, 504  # wide_hi
CROP = crop_rect(1280, 720, W / H)  # (0, 11, 1280, 698)


def check(cond: bool, msg: str) -> None:
    global FAIL
    if cond:
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


def scaler(**kw) -> GroundScaler:
    g = GroundCfg(**kw)
    return GroundScaler(g, W, H, CROP, 1280, device="cpu")


def synth(gs: GroundScaler, a_true: float, horizon_rel: float = 0.0) -> torch.Tensor:
    """땅 줄은 r = A / Z(y), 그 위(하늘)는 horizon_rel."""
    rel = torch.full((H, W), horizon_rel)
    rows, z = ground_depth_rows(H, gs.f, gs.cy, gs.gcfg.height_m, gs.gcfg.pitch_deg)
    for y, zz in zip(rows, z):
        rel[y] = a_true / zz
    return rel


print("=" * 72)
print("1. 땅 줄 깊이 기하")
print("=" * 72)
check(CROP == (0, 11, 1280, 698), f"C920 1280x720 → 924x504 크롭 {CROP}")
gs = scaler()
f, cy = gs.f, gs.cy
check(abs(f - 908 * 504 / 698) < 1e-6 and abs(cy - 252.0) < 1e-6, f"모델 좌표 f {f:.1f} px, cy {cy:.1f}")
rows, z = ground_depth_rows(H, f, cy, 0.70, 0.0)
check(abs(z[-1] - 0.70 * f / (503 - cy)) < 1e-9, f"수평: Z = f·h/(y − cy) (맨 아래 줄 {z[-1]:.3f} m)")
check(1.80 < z[-1] < 1.85, "맨 아래 줄 땅 약 1.82 m (분석과 같음)")
check(rows[0] > cy and all(a > b for a, b in zip(z, z[1:])), "수평선 아래 줄만, 아래로 갈수록 가까움")
# 기울기 5°: 땅 점 (수평 d, 높이 −h)을 카메라로 투영한 줄이 그 줄과 같아야 한다
th = math.radians(5.0)
rows5, z5 = ground_depth_rows(H, f, cy, 0.70, 5.0)
ok = True
for y, zz in list(zip(rows5, z5))[::20]:
    a = th + math.atan2(y - cy, f)
    d = 0.70 / math.tan(a)
    zc = d * math.cos(th) + 0.70 * math.sin(th)  # 광축 방향
    yc = d * math.sin(th) - 0.70 * math.cos(th)  # 광축에 수직(위 +)
    ok &= abs(cy - f * yc / zc - y) < 1e-6 and abs(zc - zz) < 1e-9
check(ok, "기울기 5°: 계산한 땅 점을 다시 투영하면 같은 줄")
check(z5[-1] < z[-1], f"아래를 보면 맨 아래 땅이 가까워짐 ({z5[-1]:.2f} < {z[-1]:.2f} m)")
check(abs(gs.z_near - z[-1]) < 1e-5 and gs.z_far <= 3.5 and len(gs.rows) >= 100,
      f"배율 계산 땅 줄 {len(gs.rows)}개, {gs.z_near:.2f}~{gs.z_far:.2f} m")
big = GroundScaler(GroundCfg(), W, H, crop_rect(1920, 1080, W / H), 1280, device="cpu")
check(abs(big.f - gs.f) < 0.5 and abs(big.z_near - gs.z_near) < 0.01,
      "원본 1920x1080이면 f를 1.5배로 맞춰 같은 기하")
try:
    scaler(pitch_deg=-25.0)
    check(False, "위를 25° 보면 땅 줄 부족 → 기동 실패")
except ValueError:
    check(True, "위를 25° 보면 땅 줄 부족 → 기동 실패")

print("\n" + "=" * 72)
print("2. 합성 상대깊이에서 배율 복원")
print("=" * 72)
for a_true in (0.5, 3.0, 40.0):
    est = gs.estimate(synth(gs, a_true))
    check(abs(est / a_true - 1) < 1e-4, f"A {a_true:g} 복원 {est:.4f}")
rel = synth(gs, 3.0)
rel[100:200, 400:500] = 3.0 / 2.5  # 2.5 m 물체
d = 3.0 / rel
check(abs(float(d[150, 450]) - 2.5) < 1e-5, "복원 배율로 물체 2.5 m")
occ = synth(gs, 3.0)
occ[gs.rows, : int(W * 0.4)] = 10.0  # 땅 줄의 40%를 가까운 물체가 가림
check(abs(gs.estimate(occ) / 3.0 - 1) < 1e-4, "땅 줄 40%가 가려져도 줄 중앙값이라 그대로")
g = torch.Generator().manual_seed(0)
noisy = synth(gs, 3.0) * (1 + 0.05 * torch.randn(H, W, generator=g))
check(abs(gs.estimate(noisy) / 3.0 - 1) < 0.02, "5% 잡음에도 2% 안")
shifted = synth(gs, 3.0, horizon_rel=0.0) + 0.3  # 이동 항이 있으면(가정 위반) 배율이 커진다
check(gs.estimate(shifted) > 3.0, "이동 항 +0.3이면 배율 과대 — r = A/Z 가정의 한계")

print("\n" + "=" * 72)
print("3. 시간 평활")
print("=" * 72)
gs = scaler(smooth_tau_s=1.0)
v0 = gs.update(synth(gs, 1.0), now=0.0)
check(abs(v0 - 1.0) < 1e-4, "첫 프레임은 그대로")
v1 = gs.update(synth(gs, 2.0), now=1.0)
want = math.exp(math.log(2.0) * (1 - math.exp(-1)))
check(abs(v1 - want) < 1e-3, f"1 τ 뒤 로그 EMA {v1:.3f} (기대 {want:.3f})")
check(abs(gs.raw - 2.0) < 1e-3, "raw는 평활 전 값")
v2 = gs.update(synth(gs, 4.0), now=7.0)
check(abs(v2 - 4.0) < 1e-3, "5 τ 넘게 쉬었다 쓰면(장면 전환 직후) 새 값으로 초기화")
v3 = gs.update(torch.zeros(H, W), now=7.1)
check(v3 == v2 and gs.n_invalid == 1, "무효 프레임(배율 0)은 직전 값 유지, 무효 수 기록")
gs.reset()
check(math.isnan(gs.update(torch.zeros(H, W), now=8.0)), "처음부터 무효면 NaN (셀 무효 처리)")
gs0 = scaler(smooth_tau_s=0.0)
gs0.update(synth(gs0, 1.0), now=0.0)
check(abs(gs0.update(synth(gs0, 2.0), now=0.1) - 2.0) < 1e-4, "smooth_tau_s 0이면 평활 안 함")

print("\n" + "=" * 72)
print("4. 엔진 출력 → 미터")
print("=" * 72)


def stub(offset=0.0, ground=None) -> DepthEngine:
    e = DepthEngine.__new__(DepthEngine)  # TRT 없이 변환만
    e.offset_m, e.ground = offset, ground
    e._depth_buf = torch.empty((H, W))
    return e


out = torch.full((H, W), 0.9)
out[0, 0] = 0.3
m = stub(offset=-0.44)._to_metres(out, None)
check(abs(float(m[1, 1]) - 0.46) < 1e-6 and abs(float(m[0, 0]) - 0.01) < 1e-6,
      "실내 −0.44 m: 0.90 → 0.46, 0.30 → 0.01(근거리 최대)")
check(abs(float(out[1, 1]) - 0.9) < 1e-6, "엔진 출력 버퍼는 바꾸지 않음")
check(stub()._to_metres(out, None) is out, "보정 0이면 출력 그대로")
gs = scaler(smooth_tau_s=0.0)
rel = synth(gs, 3.0)
rel[50, 60] = 3.0 / 4.0
m = stub(ground=gs)._to_metres(rel, 0.0)
check(abs(float(m[50, 60]) - 4.0) < 1e-4 and abs(float(m[-1, 0]) - gs.z_near) < 1e-4,
      "땅 기준: 4 m 물체, 맨 아래 땅 1.82 m")
check(float(m[0, 0]) > 1e5, "하늘(r = 0)은 아주 먼 거리 → 단계 0")
try:
    DepthEngine.__init__(DepthEngine.__new__(DepthEngine), Config().depth, Config().grid,
                         Path("/nonexistent"), offset_m=-0.4, ground=gs)
    check(False, "ground와 offset을 함께 주면 거부")
except ValueError:
    check(True, "ground와 offset을 함께 주면 거부")

print("\n" + "=" * 72)
print("5. 설정")
print("=" * 72)
base = Config()
check(base.depth.outdoor_kind == "metric" and base.depth.indoor_offset_m == 0.0,
      "설정 섹션이 없으면 예전 동작(metric, 보정 0)")
for path, val in (("depth.outdoor_kind", "relative"), ("depth.indoor_offset_m", 3.0),
                  ("ground.height_m", 0.1), ("ground.pitch_deg", 40.0), ("ground.f_px", 0.0),
                  ("ground.band_max_m", 0.8)):
    c = copy.deepcopy(base)
    sec, key = path.split(".")
    setattr(getattr(c, sec), key, val)
    try:
        c.validate()
        check(False, f"{path} = {val} 거부")
    except ValueError:
        check(True, f"{path} = {val} 거부")
c = copy.deepcopy(base)
c.depth.outdoor_kind, c.scene.mode = "ground", "off"
check(any("outdoor_kind" in w for w in c.validate()), "ground인데 scene.mode off면 경고")
cfg = load()
check(cfg.depth.outdoor_kind == "ground" and cfg.depth.indoor_offset_m == -0.44
      and cfg.depth.outdoor_engine.endswith("da2_rel_vits_924x504.engine"),
      "기본 설정: 실외 땅 기준(상대깊이 엔진), 실내 −0.44 m")
check(cfg.ground.height_m == 0.70 and cfg.ground.pitch_deg == 0.0 and cfg.scene.mode == "auto",
      "기본 설정: 높이 0.70 m, 수평, 자동 전환")
cfg.validate()
check(True, "기본 설정 검증 통과")

print()
print("FAIL 0 — 전부 통과" if not FAIL else f"FAIL {FAIL}건")
sys.exit(1 if FAIL else 0)
