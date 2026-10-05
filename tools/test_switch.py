"""
실내/실외 자동 전환 검증.

  1. 고정 모드 (indoor / outdoor)
  2. 자동 전환 — 가짜 판별기 + 가상 시각으로 결정론적으로 검사
  3. d_far 크로스페이드 (ln d_far 선형 보간)
  4. DepthMapper d_far 지정
  5. 설정 검증
  6. (선택, --with-model) 실제 엔진·판별기로 Pipeline: 실내 사진 → 실외 사진 → 실내 사진

  python tools/test_switch.py
  python tools/test_switch.py --with-model
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision.config import Config, FilterCfg, MappingCfg, SceneCfg, load  # noqa: E402
from tactile_vision.mapping import DepthMapper  # noqa: E402
from tactile_vision.scene import SceneResult  # noqa: E402
from tactile_vision.switching import SceneSwitcher  # noqa: E402

EXAMPLES = Path("/home/june/sensory_substitution/Depth-Anything-V2/assets/examples")
FAIL = 0


def check(cond: bool, msg: str) -> None:
    global FAIL
    if cond:
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


class FakeClassifier:
    """P(실외)를 스크립트로 정하는 판별기. 호출 횟수를 센다."""

    def __init__(self) -> None:
        self.p = 0.05
        self.calls = 0

    def classify(self, bgr: object) -> SceneResult:
        self.calls += 1
        return SceneResult(self.p, [("fake", 1.0, self.p > 0.5)], 1.0)


def make(**kw: object) -> tuple[SceneSwitcher, FakeClassifier]:
    fake = FakeClassifier()
    return SceneSwitcher(SceneCfg(**{"mode": "auto", **kw}), fake), fake


def run_seq(sw: SceneSwitcher, fake: FakeClassifier, seq: list[tuple[float, float]],
            fps: float = 30.0, t0: float = 0.0) -> list[tuple[float, object, int]]:
    """seq = [(지속 초, P(실외)), ...] → [(시각, 결정, 구간 번호)]. 시각은 인덱스로 계산(누적 오차 없음)."""
    out = []
    i = 0
    for k, (dur, p) in enumerate(seq):
        fake.p = p
        for _ in range(int(round(dur * fps))):
            t = t0 + i / fps
            out.append((t, sw.step(None, now=t), k))
            i += 1
    return out


def switches(log: list) -> list[tuple[float, str]]:
    return [(t, d.state) for (t, d, _), (_, dp, _) in zip(log[1:], log[:-1]) if d.state != dp.state]


parser = argparse.ArgumentParser()
parser.add_argument("--with-model", action="store_true", help="GPU로 실제 엔진·판별기까지 검사")
args = parser.parse_args()

print("=" * 72)
print("1. 고정 모드 (판별 안 함)")
print("=" * 72)
for mode, far in (("indoor", 4.0), ("outdoor", 8.0)):
    sw = SceneSwitcher(SceneCfg(mode=mode))
    ds = [sw.step(None, now=i / 30) for i in range(60)]
    check(all(d.models == (mode,) and d.far_m == far and d.state == mode for d in ds),
          f"{mode}: 항상 {mode} 모델 하나, d_far {far} m")
    check(all(not d.uncertain and d.p_out is None and not d.classified for d in ds),
          f"{mode}: 판별 안 함, 애매 표시 없음")
for bad in ({"mode": "auto"}, {"mode": "off"}):
    try:
        SceneSwitcher(SceneCfg(**bad))
        check(False, f"거부: {bad} (판별기 없음 / off)")
    except ValueError:
        check(True, f"거부: {bad} (판별기 없음 / off)")

print("\n" + "=" * 72)
print("2. 자동 전환 (가짜 판별기, 가상 30 fps)")
print("=" * 72)
sw, fake = make()
log = run_seq(sw, fake, [(4, 0.05), (5, 0.95), (5, 0.05)])
seg = lambda k: [d for _, d, kk in log if kk == k]  # noqa: E731
check(all(d.state == "indoor" and d.models == ("indoor",) and d.far_m == 4.0 for d in seg(0)),
      "실내 4초: 실내 모델 하나, d_far 4 m")
sws = switches(log)
check([s for _, s in sws] == ["outdoor", "indoor"], f"전환 2회: 실외 → 실내 (실제 {sws})")
lat_out = sws[0][0] - 4.0 if sws else float("nan")
lat_in = sws[1][0] - 9.0 if len(sws) > 1 else float("nan")
# 평활 τ=1 s로 0.05→0.95가 0.65를 넘는 데 ln3 ≈ 1.10 s + 체류 1.5 s ≈ 2.6 s (판별 간격 0.2 s 오차)
check(2.4 <= lat_out <= 3.0 and 2.4 <= lat_in <= 3.0,
      f"장면이 바뀐 뒤 전환까지 2.4~3.0 s (실외 {lat_out:.2f}s / 실내 {lat_in:.2f}s)")
pend = [d for t, d, k in log if k == 1 and t < sws[0][0]]
both_before = [d for d in pend if d.models_label == "both"]
check(len(both_before) > 0 and all(d.state == "indoor" for d in both_before),
      f"전환 전 애매 구간에서 두 모델 사용 ({len(both_before) / 30:.1f}s), 확정 장면은 실내 유지")
check(all(d.far_m == 4.0 for d in pend), "애매 구간에도 d_far는 확정 장면(실내 4 m) 유지")
tail_out = [d for t, d, k in log if k == 1 and t > sws[0][0] + 0.8]
check(len(tail_out) > 0 and all(d.models == ("outdoor",) and d.far_m == 8.0 for d in tail_out),
      "전환 후 안정 구간: 실외 모델 하나, d_far 8 m")
tail_in = [d for t, d, k in log if k == 2 and t > sws[1][0] + 0.8]
check(len(tail_in) > 0 and all(d.models == ("indoor",) and d.far_m == 4.0 for d in tail_in),
      "되돌아온 뒤: 실내 모델 하나, d_far 4 m")

rates = {}
for fps in (15.0, 30.0, 60.0):
    s2, f2 = make()
    run_seq(s2, f2, [(10, 0.05)], fps=fps)
    rates[fps] = f2.calls / 10.0
check(all(abs(r - 5.0) <= 0.15 for r in rates.values()),
      "판별 횟수는 프레임률과 무관하게 초당 5회 ("
      + ", ".join(f"{k:g}fps → {v:.1f}Hz" for k, v in rates.items()) + ")")

s2, f2 = make()
lg = run_seq(s2, f2, [(4, 0.05), (1.0, 0.95), (5, 0.05)])
check(not switches(lg), "1초짜리 실외 판정(문 앞 등) → 전환 없음")
check(any(d.models_label == "both" for _, d, _ in lg), "그 사이 애매해진 동안에는 두 모델로 가까운 값 사용")

s2, f2 = make(uncertain_policy="hold")
lg = run_seq(s2, f2, [(4, 0.05), (5, 0.95), (5, 0.05)])
check(all(len(d.models) == 1 and d.models[0] == d.state for _, d, _ in lg),
      "uncertain_policy=hold: 애매해도 확정 장면 모델 하나만")
check([s for _, s in switches(lg)] == ["outdoor", "indoor"], "hold도 전환 시점은 같음")

s2, f2 = make()
f2.p = 0.55
d0 = s2.step(None, now=0.0)
check(d0.state == "outdoor" and d0.uncertain and d0.models_label == "both",
      "첫 판정이 애매(55%): 0.5 기준으로 실외, 두 모델 사용")

print("\n" + "=" * 72)
print("3. d_far 크로스페이드 (ln d_far를 0.7 s 동안 선형 보간)")
print("=" * 72)
t_sw = sws[0][0]
fade = [(t - t_sw, d.far_m) for t, d, k in log if k == 1 and t >= t_sw]
mid = min(fade, key=lambda x: abs(x[0] - 0.35))
check(abs(mid[1] - math.exp(math.log(4) + (mid[0] / 0.7) * math.log(2))) < 1e-6,
      f"보간 중 값 = exp(ln4 + a·ln2), t+{mid[0]:.2f}s → {mid[1]:.2f} m (기하평균 5.66 근처)")
during = [f for dt, f in fade if dt < 0.7]
check(all(b >= a for a, b in zip(during, during[1:])) and during[0] == 4.0,
      "4 → 8 m 단조 증가, 전환 프레임에서 4 m부터 시작 (튀지 않음)")
check(all(f == 8.0 for dt, f in fade if dt >= 0.7), "0.7 s 이후 정확히 8 m")

s3, f3 = make()
f3.p = 0.95
d = s3.step(None, now=0.0)
check(d.state == "outdoor" and d.far_m == 8.0, "첫 판정은 보간 없이 바로 목표 d_far (실외 8 m)")
# 보간 도중 반대로 전환: 지금 값에서 새 목표로 이어서 보간
s4, f4 = make(dwell_s=0.0, tau_s=0.0)
lg = run_seq(s4, f4, [(1, 0.05), (0.3, 0.95), (2, 0.05)])
fs = [d.far_m for _, d, _ in lg]
jumps = max(abs(b - a) for a, b in zip(fs, fs[1:]))
check([s for _, s in switches(lg)] == ["outdoor", "indoor"] and fs[-1] == 4.0,
      "보간 도중(0.3 s) 되돌아감 → 다시 4 m로 수렴")
check(jumps < 0.5, f"되돌아가는 순간에도 d_far가 튀지 않음 (프레임 간 최대 변화 {jumps:.2f} m)")
s5, f5 = make(crossfade_s=0.0)
lg = run_seq(s5, f5, [(4, 0.05), (5, 0.95)])
t5 = switches(lg)[0][0]
check(all(d.far_m == (8.0 if t >= t5 else 4.0) for t, d, _ in lg), "crossfade_s=0이면 즉시 바뀜")

print("\n" + "=" * 72)
print("4. DepthMapper d_far 지정")
print("=" * 72)
FAST = FilterCfg(ema_alpha_approach=1.0, ema_alpha_recede=1.0, hysteresis_up=0.0,
                 hysteresis_down=0.0, confirm_frames_up=1, confirm_frames_down=1)
MC = MappingCfg(near_m=0.5, far_m=8.0, levels=8, stroke_mm=geo.STROKE_MM_DEFAULT)


def level_at(d: float, far: float | None, cfg: MappingCfg = MC, n: int = 1) -> int:
    mp = DepthMapper(cfg, FAST, geo.CELL_COUNT)
    mp.set_far(far)
    lv = None
    for _ in range(n):
        lv = mp.update(np.full(geo.CELL_COUNT, d, np.float32))
    return int(lv[0])


check(level_at(4.5, 4.0) == 0 and level_at(4.5, 8.0) > 0,
      f"4.5 m: d_far 4 m면 레벨 0, 8 m면 {level_at(4.5, 8.0)} (먼 쪽까지 표현)")
want = round(math.log(4.0 / 1.0) / math.log(4.0 / 0.5) * 7)
check(level_at(1.0, 4.0) == want, f"1.0 m, d_far 4 m → 로그 매핑 레벨 {want}")
mp = DepthMapper(MC, FAST, geo.CELL_COUNT)
mp.set_far(4.0)
mp.reset()
check(mp.active_far_m == 4.0, "reset() 후에도 지정한 d_far 유지")
mp.set_far(None)
check(mp.active_far_m == 8.0, "set_far(None) → mapping.far_m(8 m)으로 복귀")
try:
    mp.set_far(0.4)
    check(False, "d_far ≤ near 거부")
except ValueError:
    check(True, "d_far ≤ near 거부")
mp = DepthMapper(MC, FAST, geo.CELL_COUNT)
mp.set_far(4.0)
lv = mp.update(np.full(geo.CELL_COUNT, np.nan, np.float32))
check(int(lv.max()) == 0, "무효값(NaN)은 d_far 4 m에서도 핀을 올리지 않음")
ad = MappingCfg(**{**MC.__dict__, "adaptive_far": True, "adaptive_far_max_m": 15.0})
mp = DepthMapper(ad, FAST, geo.CELL_COUNT)
mp.set_far(4.0)
for _ in range(60):
    mp.update(np.full(geo.CELL_COUNT, 3.0, np.float32))  # 경로에 물체 있음 → 기준 d_far로
far_busy = mp.active_far_m
for _ in range(60):
    mp.update(np.full(geo.CELL_COUNT, 5.0, np.float32))  # 전부 d_far 밖 → 확장
far_clear = mp.active_far_m
check(abs(far_busy - 4.0) < 0.05 and abs(far_clear - 8.0) < 0.05,
      f"adaptive_far는 지정 d_far 기준: 물체 있음 {far_busy:.2f} m(→4), "
      f"경로 클리어 {far_clear:.2f} m(→5×1.6=8)")

print("\n" + "=" * 72)
print("5. 설정 검증")
print("=" * 72)
cfg = load()
check(cfg.scene.mode == "auto", f"default.yaml scene.mode = auto (실제 {cfg.scene.mode})")
check(Config().scene.mode == "off", "scene 섹션 없는 설정은 off (예전 동작)")
try:
    cfg.validate()
    check(True, "default.yaml 검증 통과")
except ValueError as e:
    check(False, f"default.yaml 검증 통과 ({e})")
for bad in ({"mode": "cave"}, {"enter_indoor": 0.7, "enter_outdoor": 0.6},
            {"far_indoor_m": 0.5}, {"uncertain_policy": "maybe"}, {"rate_hz": 0.0},
            {"crossfade_s": -1.0}):
    c2 = copy.deepcopy(cfg)
    for k, v in bad.items():
        setattr(c2.scene, k, v)
    try:
        c2.validate()
        check(False, f"거부: scene {bad}")
    except ValueError:
        check(True, f"거부: scene {bad}")

if args.with_model:
    print("\n" + "=" * 72)
    print("6. 실제 엔진·판별기 (GPU): 실내 4s → 실외 5s → 실내 5s, 실시간")
    print("=" * 72)
    import cv2

    from tactile_vision.camera import Frame, crop_rect
    from tactile_vision.pipeline import Pipeline

    cfg = load()
    spec = cfg.depth.spec

    def prep(name: str) -> np.ndarray:
        img = cv2.imread(str(EXAMPLES / name))
        x, y, w, h = crop_rect(img.shape[1], img.shape[0], spec.width / spec.height)
        return cv2.resize(img[y:y + h, x:x + w], (spec.width, spec.height),
                          interpolation=cv2.INTER_AREA)

    class ScriptSource:
        """구간별로 정지 사진을 내보내는 소스. 시각은 실제 시간(카메라와 같은 기준)."""

        def __init__(self, segs: list[tuple[float, np.ndarray]]) -> None:
            self.segs, self.t0, self.n = segs, None, 0
            self.seg_of: dict[int, int] = {}
            self.t_of: dict[int, float] = {}

        def read(self, timeout: float = 1.0) -> Frame | None:
            now = time.perf_counter()
            self.t0 = now if self.t0 is None else self.t0
            acc = 0.0
            for k, (dur, img) in enumerate(self.segs):
                acc += dur
                if now - self.t0 < acc:
                    self.n += 1
                    self.seg_of[self.n], self.t_of[self.n] = k, now - self.t0
                    return Frame(image=img, raw=img, ts=now, index=self.n)
            return None

        def close(self) -> None:
            pass

    ind_img, out_img = prep("demo15.jpg"), prep("demo01.jpg")  # 현관 홀 / 도심 거리

    def run_segments(policy: str):
        """실내 4 s → 실외 5 s → 실내 5 s를 실시간으로. 애매 구간 정책만 바꿔 같은 시험을 한다."""
        import dataclasses
        import gc

        import torch

        c = dataclasses.replace(cfg, scene=dataclasses.replace(cfg.scene, uncertain_policy=policy),
                                runtime=dataclasses.replace(cfg.runtime))
        with tempfile.TemporaryDirectory() as td:
            c.runtime.metrics_csv = str(Path(td) / "metrics.csv")  # logs/metrics.csv 덮어쓰지 않음
            src = ScriptSource([(4.0, ind_img), (5.0, out_img), (5.0, ind_img)])
            pipe = Pipeline(c, src, None)
            pipe.warmup(10)
            frames, both_chk = [], None  # (셀별 최소와 일치?, 실외/실내 셀 거리 비)
            while (r := pipe.step()) is not None:
                frames.append(r)
                if both_chk is None and r.scene.models_label == "both":
                    a = pipe.engines["indoor"].infer(r.frame.image)
                    b = pipe.engines["outdoor"].infer(r.frame.image)
                    both_chk = (bool(np.allclose(r.cell_dist, np.fmin(a, b), atol=1e-3)),
                                float(np.mean(b / a)))
            single = [r for r in frames if src.seg_of[r.frame.index] == 1][-5:]
            ok_single = all(np.allclose(r.cell_dist, pipe.engines[r.scene.models[0]].infer(r.frame.image),
                                        atol=1e-3) for r in single if len(r.scene.models) == 1)
            dm_shape = pipe.last_depth_map().shape
            loaded = set(pipe.engines) == {"indoor", "outdoor"} and pipe.classifier is not None
            pipe.close()
            with open(c.runtime.metrics_csv, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return frames, rows, both_chk, src, ok_single, dm_shape, loaded

    def common_checks(tag, frames, rows, src, ok_single, dm_shape, loaded):
        seg_r = [[r for r in frames if src.seg_of[r.frame.index] == k] for k in range(3)]
        st = [(src.t_of[r.frame.index], r.scene.state) for r in frames]
        sw_t = [(t, s) for (t, s), (_, sp) in zip(st[1:], st[:-1]) if s != sp]
        print(f"  [{tag}] 프레임 {len(frames)}개, 전환 {[(round(t, 2), s) for t, s in sw_t]}")
        check(loaded, f"[{tag}] auto: 실내·실외 엔진과 판별기 로드")
        check(all(r.scene.state == "indoor" for r in seg_r[0]), f"[{tag}] 실내 사진 구간: 실내로 판정")
        check([s for _, s in sw_t] == ["outdoor", "indoor"],
              f"[{tag}] 실외 사진에서 실외로, 다시 실내 사진에서 실내로")
        if len(sw_t) == 2:
            l1, l2 = sw_t[0][0] - 4.0, sw_t[1][0] - 9.0
            check(2.2 <= l1 <= 3.5 and 2.2 <= l2 <= 3.5,
                  f"[{tag}] 사진이 바뀐 뒤 전환까지 2.2~3.5 s (실외 {l1:.2f}s / 실내 {l2:.2f}s)")
        check(ok_single and seg_r[1][-1].scene.models == ("outdoor",),
              f"[{tag}] 실외 확정 후: 실외 모델 하나의 셀 거리 그대로")
        check(abs(seg_r[1][-1].scene.far_m - 8.0) < 1e-9 and abs(seg_r[2][-1].scene.far_m - 4.0) < 1e-9,
              f"[{tag}] d_far: 실외 구간 끝 8 m, 다시 실내 구간 끝 4 m")
        clf = [r.times.scene for r in frames if r.scene.classified]
        noclf = [r.times.scene for r in frames if not r.scene.classified]
        print(f"        장면 판별 프레임 {len(clf)}개 ({len(clf) / 14:.1f}회/초) "
              f"중앙값 {np.median(clf):.1f}ms, 판별 없는 프레임 {np.median(noclf):.2f}ms")
        tot = [r.times.total for r in frames]
        print(f"        프레임 처리 중앙값 {np.median(tot):.1f}ms ({len(frames) / 14:.1f} 프레임/초), "
              f"p95 {np.percentile(tot, 95):.1f}ms")
        need = {"scene_ms", "scene", "uncertain", "p_out", "p_smooth", "models", "far_m", "d0", "L23"}
        check(len(rows) == len(frames) and need <= set(rows[0]),
              f"[{tag}] CSV: 프레임마다 1행, 장면 열 포함 ({len(rows)}행)")
        check(dm_shape == (spec.height, spec.width), f"[{tag}] last_depth_map {dm_shape}")

    # 기본 설정(hold): 애매해도 모델 하나
    frames, rows, _, src, ok_single, dm_shape, loaded = run_segments("hold")
    common_checks("hold", frames, rows, src, ok_single, dm_shape, loaded)
    unc = [r for r in frames if r.scene.uncertain]
    check(unc and all(len(r.scene.models) == 1 and r.scene.models[0] == r.scene.state for r in frames),
          f"[hold] 두 모델 프레임 0개 — 애매 구간 {len(unc)}프레임도 확정 장면 모델 하나")
    check({r["models"] for r in rows} == {"indoor", "outdoor"}, "[hold] CSV models 값: indoor·outdoor")

    # both: 애매하면 두 모델의 셀별 가까운 값
    frames, rows, both_chk, src, ok_single, dm_shape, loaded = run_segments("both")
    common_checks("both", frames, rows, src, ok_single, dm_shape, loaded)
    check(both_chk is not None and both_chk[0],
          "[both] 두 모델 프레임의 셀 거리 = 두 모델 셀별 최소 (직접 재계산과 일치)")
    if both_chk:
        print(f"        (같은 장면에서 실외/실내 셀 거리 비 평균 {both_chk[1]:.2f}배)")
    one = [r.times.depth for r in frames if len(r.scene.models) == 1]
    two = [r.times.depth for r in frames if len(r.scene.models) == 2]
    ratio = np.median(two) / np.median(one) if one and two else float("nan")
    check(1.6 <= ratio <= 2.4, f"[both] 깊이 시간: 모델 1개 {np.median(one):.1f}ms, 2개 {np.median(two):.1f}ms "
          f"({ratio:.2f}배)")
    check({r["models"] for r in rows} == {"indoor", "outdoor", "both"}
          and {r["scene"] for r in rows} == {"indoor", "outdoor"},
          "[both] CSV models/scene 값: indoor·outdoor·both / indoor·outdoor")

print()
print("FAIL 0 — 전부 통과" if not FAIL else f"FAIL {FAIL}건")
sys.exit(1 if FAIL else 0)
