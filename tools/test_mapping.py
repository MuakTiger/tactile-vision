"""매핑 + 비대칭 시간필터 검증. 설계 주장을 수치로 확인한다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision.config import FilterCfg, MappingCfg  # noqa: E402
from tactile_vision.mapping import DepthMapper, level_edges  # noqa: E402

FAIL = 0
N_CELLS = geo.CELL_COUNT
FPS = 15
WARMUP = 45  # 정상상태 측정을 위해 버리는 초기 프레임 수
BASE = MappingCfg(near_m=0.5, far_m=8.0, levels=8, stroke_mm=geo.STROKE_MM_DEFAULT)
STEP_MM = geo.DEFAULT_PROFILE.step_mm  # stroke/(levels-1)


def check(cond: bool, msg: str) -> None:
    global FAIL
    if cond:
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


def cfg_of(mode: str) -> MappingCfg:
    return MappingCfg(**{**BASE.__dict__, "mode": mode})


def run(mp: DepthMapper, series: np.ndarray) -> np.ndarray:
    return np.array([mp.update(series[i], dt=1 / FPS) for i in range(len(series))])


def moves_of(seq: np.ndarray) -> tuple[int, float]:
    d = np.diff(seq, axis=0)
    return int(np.count_nonzero(d)), float(np.abs(d).sum() * STEP_MM)


print("=" * 72)
print("1. 매핑 모드별 단계 경계 (8단계, 0.5~8m)")
print("=" * 72)
for mode in ("linear", "log", "disparity"):
    e = level_edges(cfg_of(mode))
    n_crit = int(np.sum((e[:-1] >= 0.5) & (e[:-1] < 1.0)))
    print(f"\n  [{mode:9s}] " + "  ".join(f"{v:5.2f}" for v in e))
    print(f"              0.5~1.0m 단계수 {n_crit} | 최원거리 단계폭 {e[-1]-e[-2]:.2f}m")

e_log, e_lin, e_dis = (level_edges(cfg_of(m)) for m in ("log", "linear", "disparity"))
check(
    np.sum((e_log[:-1] >= 0.5) & (e_log[:-1] < 1.0))
    > np.sum((e_lin[:-1] >= 0.5) & (e_lin[:-1] < 1.0)),
    "log가 충돌구간(0.5~1m)에 선형보다 많은 단계 배정",
)
check(
    (e_log[-1] - e_log[-2]) > (e_lin[-1] - e_lin[-2]),
    "log가 원거리를 포화 대신 압축 (최원거리 단계폭이 더 큼)",
)
check(
    (e_dis[-1] - e_dis[-2]) > (e_log[-1] - e_log[-2]) * 2,
    "disparity는 원거리 단계가 과도하게 넓음 (log 대비 2배 초과) → 부적합",
)

print("\n" + "=" * 72)
print("2. 정상상태 떨림 — 정지한 벽 앞 노이즈 (2.0m ± 5cm, 300프레임)")
print("=" * 72)
rng = np.random.default_rng(42)
noisy = 2.0 + rng.normal(0, 0.05, size=(300, N_CELLS))

variants = [
    ("필터 없음", FilterCfg(1.0, 1.0, 0.0, 0.0, 1, 1)),
    ("대칭 EMA만", FilterCfg(0.35, 0.35, 0.0, 0.0, 1, 1)),
    ("비대칭 EMA", FilterCfg(0.65, 0.25, 0.0, 0.0, 1, 1)),
    ("전체(기본설정)", FilterCfg(0.65, 0.25, 0.15, 0.45, 1, 3)),
]
results = {}
for label, fcfg in variants:
    seq = run(DepthMapper(cfg_of("log"), fcfg, N_CELLS), noisy)
    mv, dist = moves_of(seq[WARMUP:])
    per_min = mv / N_CELLS / ((len(seq) - WARMUP) / FPS) * 60
    results[label] = mv
    print(
        f"  {label:<16} 이동 {mv:5d}회  총 {dist:8.1f}mm  "
        f"서보당 {per_min:7.1f}회/분"
    )

# 2.0m는 log 매핑의 레벨 경계와 정확히 일치하는 최악 조건이다.
# 경계 위에 신호가 놓이면 히스테리시스로도 간헐적 교차를 완전히 막을 수 없다.
# 실용 기준: 서보당 분당 1회 미만.
worst = results["전체(기본설정)"] / N_CELLS / ((300 - WARMUP) / FPS) * 60
check(worst < 1.0, f"기본설정 떨림이 서보당 분당 1회 미만 (실측 {worst:.2f}회/분)")
check(
    results["전체(기본설정)"] < results["필터 없음"] * 0.01,
    f"필터 없음 대비 99% 이상 감소 ({results['필터 없음']}→{results['전체(기본설정)']})",
)

print("\n" + "=" * 72)
print("3. 접근 응답성 — 4.0m → 0.5m, 1.0 m/s, 노이즈 ±5cm")
print("=" * 72)
appr = np.arange(4.0, 0.49, -1.0 / FPS)
settle = np.full(20, appr[-1])  # 도달 후 정착 구간
series = np.concatenate([appr, settle])
noisy_appr = series[:, None] + rng.normal(0, 0.05, size=(len(series), N_CELLS))

for label, fcfg in [
    ("대칭 필터", FilterCfg(0.35, 0.35, 0.25, 0.25, 2, 2)),
    ("비대칭 필터(기본)", FilterCfg(0.65, 0.25, 0.15, 0.45, 1, 3)),
]:
    mp = DepthMapper(cfg_of("log"), fcfg, N_CELLS)
    seq = run(mp, noisy_appr)
    lv = seq[:, 0]
    tr = np.nonzero(np.diff(lv))[0]
    # 각 전이가 이상 경계보다 얼마나 늦었는지
    lag = [series[i] - e_log[8 - lv[i + 1]] for i in tr if lv[i + 1] <= 7]
    print(f"\n  [{label}]")
    print(f"    최종 레벨 {lv[-1]} (목표 7) | 전이 {len(tr)}회")
    print(f"    전이 거리: {[f'{series[i]:.2f}' for i in tr]}")
    print(f"    경계 지연: 평균 {np.mean(lag)*100:+.1f}cm")
    if label == "대칭 필터":
        sym_lag, sym_final = float(np.mean(lag)), int(lv[-1])
    else:
        asym_lag, asym_final = float(np.mean(lag)), int(lv[-1])

check(asym_final == 7, f"비대칭 필터가 최대 레벨 7 도달 (실측 {asym_final})")
check(
    abs(asym_lag) < abs(sym_lag),
    f"비대칭이 접근 지연을 줄임 ({abs(sym_lag)*100:.0f}cm → {abs(asym_lag)*100:.0f}cm)",
)
check(abs(asym_lag) < 0.15, f"접근 지연 15cm 이내 (실측 {abs(asym_lag)*100:.0f}cm)")

print("\n" + "=" * 72)
print("4. 후퇴 둔감성 — 장애물이 갑자기 사라져도 핀이 천천히 내려가는가")
print("=" * 72)
recede = np.concatenate([np.full(30, 0.8), np.full(60, 6.0)])
noisy_rec = recede[:, None] + rng.normal(0, 0.05, size=(len(recede), N_CELLS))
mp = DepthMapper(cfg_of("log"), FilterCfg(0.65, 0.25, 0.15, 0.45, 1, 3), N_CELLS)
seq = run(mp, noisy_rec)
lv = seq[:, 0]
after = lv[30:]
drop_frames = int(np.argmax(after <= 1)) if np.any(after <= 1) else -1
print(f"  0.8m 유지 중 레벨: {lv[29]}")
print(f"  6.0m로 전환 후 레벨 1 이하까지: {drop_frames}프레임 "
      f"({drop_frames/FPS*1000:.0f}ms)")
print(f"  최종 레벨: {lv[-1]}")
check(drop_frames > 0, "후퇴가 결국 반영됨 (멈춰있지 않음)")
check(drop_frames >= 5, f"후퇴 반영이 점진적 ({drop_frames}프레임, 급락 아님)")

print("\n" + "=" * 72)
print("5. 한 프레임 다단계 점프 — 상승은 허용, 하강은 1단계 제한")
print("=" * 72)
jump_in = np.concatenate([np.full(20, 7.0), np.full(10, 0.55)])
mp = DepthMapper(cfg_of("log"), FilterCfg(0.65, 0.25, 0.15, 0.45, 1, 3), N_CELLS)
lv_in = run(mp, jump_in[:, None].repeat(N_CELLS, 1))[:, 0]
rise = int(np.max(np.diff(lv_in[19:])))
print(f"  7.0m → 0.55m 급접근: 레벨 {lv_in[19]} → {lv_in[-1]}, 최대 1프레임 상승폭 {rise}")

jump_out = np.concatenate([np.full(30, 0.55), np.full(40, 7.0)])
mp = DepthMapper(cfg_of("log"), FilterCfg(0.65, 0.25, 0.15, 0.45, 1, 3), N_CELLS)
lv_out = run(mp, jump_out[:, None].repeat(N_CELLS, 1))[:, 0]
fall = int(np.min(np.diff(lv_out[29:])))
print(f"  0.55m → 7.0m 급후퇴: 레벨 {lv_out[29]} → {lv_out[-1]}, 최대 1프레임 하강폭 {abs(fall)}")
check(rise > 1, f"급접근 시 즉시 다단계 상승 ({rise}단계) — 안전 반응")
check(abs(fall) == 1, f"급후퇴 시 1단계씩만 하강 (실측 {abs(fall)}) — 급락 방지")

print("\n" + "=" * 72)
print("6. metric+log vs relative_ema — 같은 핀 높이의 의미 일관성")
print("=" * 72)
scenes = {
    "복도(0.5~10m)": np.linspace(0.5, 10.0, N_CELLS),
    "벽앞(1.8~2.2m)": np.linspace(1.8, 2.2, N_CELLS),
    "로비(3~15m)": np.linspace(3.0, 15.0, N_CELLS),
}
spreads = {}
for mode in ("log", "relative_ema"):
    print(f"\n  [{mode}] 레벨 4가 뜻하는 실제 거리:")
    vals = []
    for name, dists in scenes.items():
        mp = DepthMapper(cfg_of(mode), FilterCfg(1.0, 1.0, 0.0, 0.0, 1, 1), N_CELLS)
        out = run(mp, np.tile(dists, (150, 1)))[-1]
        idx = np.nonzero(out == 4)[0]
        got = float(np.mean(dists[idx])) if idx.size else float("nan")
        vals.append(got)
        print(f"    {name:<16} {got:6.2f}m" + ("   (해당 레벨 없음)" if not idx.size else ""))
    fin = [v for v in vals if np.isfinite(v)]
    spreads[mode] = max(fin) / min(fin) if len(fin) > 1 else 1.0
    print(f"    → 장면간 배율 {spreads[mode]:.2f}x")

check(spreads["log"] < 1.5, f"metric+log는 의미 유지 ({spreads['log']:.2f}x)")
check(
    spreads["relative_ema"] > 3.0,
    f"relative_ema는 의미가 장면따라 크게 변동 ({spreads['relative_ema']:.2f}x)",
)

print("\n" + "=" * 72)
print(f"결과: {'전체 통과' if FAIL == 0 else f'{FAIL}개 실패'}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
