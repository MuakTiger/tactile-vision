"""
실내/실외 판별 검증.

  1. 라벨 파일 정합성 (365개, 순서, 보행 관련 장면의 실내/실외 표시)
  2. 가중치 무결성 (sha256)
  3. SceneGate 전환 동작 — 합성 시계열, 가상 시각으로 결정론적으로 검사
  4. (선택, --with-model) 실제 모델로 예제 사진 판정 + 속도

  python tools/test_scene.py
  python tools/test_scene.py --with-model
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision.scene import (  # noqa: E402
    CATEGORIES_FILE,
    IO_FILE,
    WEIGHTS_FILE,
    WEIGHTS_SHA256,
    SceneGate,
    _sha256,
    load_labels,
)

CK = REPO / "checkpoints" / "places365"
EXAMPLES = Path("/home/june/sensory_substitution/Depth-Anything-V2/assets/examples")
FAIL = 0


def check(cond: bool, msg: str) -> None:
    global FAIL
    if cond:
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


def feed(gate: SceneGate, seq: list[tuple[float, float]], hz: float = 5.0,
         t0: float = 0.0) -> tuple[float, list[tuple[float, str]]]:
    """
    seq = [(지속 초, P(실외)), ...]를 hz 주기로 관측시킨다.
    반환: (끝 시각, [(전환 시각, 새 상태), ...]). 시각은 누적 오차 없이 인덱스로 계산.
    """
    dt = 1.0 / hz
    i = 0
    prev = gate.state
    switches: list[tuple[float, str]] = []
    for dur, p in seq:
        for _ in range(int(round(dur * hz))):
            t = t0 + i * dt
            s = gate.observe(p, now=t)
            if prev is not None and s != prev:
                switches.append((t, s))
            prev = s
            i += 1
    return t0 + i * dt, switches


parser = argparse.ArgumentParser()
parser.add_argument("--with-model", action="store_true", help="GPU로 실제 모델까지 검사")
args = parser.parse_args()

print("=" * 72)
print("1. 라벨 파일 (공식 IO_places365.txt: 1=실내, 2=실외)")
print("=" * 72)
names, outdoor = load_labels(CK)
check(len(names) == 365 and outdoor.shape == (365,), f"장면 365개 (실제 {len(names)})")
n_in, n_out = int((~outdoor).sum()), int(outdoor.sum())
check((n_in, n_out) == (161, 204), f"실내 161 / 실외 204 (실제 {n_in} / {n_out})")
lab = dict(zip(names, outdoor))
walk_in = ["corridor", "lobby", "elevator_lobby", "staircase", "entrance_hall",
           "shopping_mall/indoor", "subway_station/platform", "parking_garage/indoor"]
walk_out = ["street", "crosswalk", "parking_lot", "alley", "downtown", "plaza", "park",
            "residential_neighborhood", "parking_garage/outdoor"]
check(not any(bool(lab[n]) for n in walk_in), "보행 실내 장면 8종이 실내로 표시")
check(all(bool(lab[n]) for n in walk_out), "보행 실외 장면 9종이 실외로 표시")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    cats = (CK / CATEGORIES_FILE).read_text().splitlines()
    ios = (CK / IO_FILE).read_text().splitlines()
    (tdp / CATEGORIES_FILE).write_text("\n".join(cats) + "\n")
    ios[0], ios[1] = ios[1], ios[0]  # 두 줄 순서를 바꾼 손상 파일
    (tdp / IO_FILE).write_text("\n".join(ios) + "\n")
    try:
        load_labels(tdp)
        check(False, "순서가 어긋난 라벨 파일을 거부")
    except RuntimeError:
        check(True, "순서가 어긋난 라벨 파일을 거부")

print("\n" + "=" * 72)
print("2. 가중치 무결성")
print("=" * 72)
w = CK / WEIGHTS_FILE
if w.exists():
    check(_sha256(w) == WEIGHTS_SHA256, f"{WEIGHTS_FILE} sha256 = 검증된 값")
else:
    check(False, f"가중치 없음: {w}")

print("\n" + "=" * 72)
print("3. SceneGate 전환 (기본값: tau 1.0s, 문턱 65%/35%, 유지 1.5s, 관측 5Hz)")
print("=" * 72)

g = SceneGate()
g.observe(0.9, now=0.0)
check(g.state == "outdoor", "첫 관측 90% → 실외로 시작")
g = SceneGate()
g.observe(0.2, now=0.0)
check(g.state == "indoor", "첫 관측 20% → 실내로 시작")
g = SceneGate(initial="outdoor")
g.observe(0.1, now=0.0)
check(g.state == "outdoor", "initial 지정 시 첫 관측이 덮어쓰지 않음")

# 깨끗한 전환: 실내 5초 → 실외 10초
g = SceneGate()
_, sw = feed(g, [(5, 0.05), (10, 0.95)])
lat = sw[0][0] - 5.0 if sw else float("nan")
check(len(sw) == 1 and sw[0][1] == "outdoor",
      f"실내→실외 계단 입력에 정확히 1회 전환 (실제 {len(sw)}회)")
check(2.4 <= lat <= 3.2, f"전환 지연 2.4~3.2초 (평활 1.1초 + 유지 1.5초, 실제 {lat:.1f}초)")
check(not g.uncertain, "전환 후 안정 상태에서는 uncertain 아님")

# 반대 방향
g = SceneGate()
_, sw = feed(g, [(5, 0.95), (10, 0.05)])
lat = sw[0][0] - 5.0 if sw else float("nan")
check(len(sw) == 1 and sw[0][1] == "indoor" and 2.4 <= lat <= 3.2,
      f"실외→실내도 대칭 (전환 {len(sw)}회, 지연 {lat:.1f}초)")

# 출입문: 프레임마다 판정이 뒤집힘
g = SceneGate()
_, sw = feed(g, [(0.2, 0.1), (0.2, 0.9)] * 50)
check(len(sw) == 0 and g.uncertain,
      f"0.2초마다 10%↔90% 요동 20초 → 전환 0회, uncertain (실제 {len(sw)}회)")

# 문을 열었다 닫았다: 1초 간격
g = SceneGate()
_, sw = feed(g, [(1.0, 0.1), (1.0, 0.9)] * 10)
check(len(sw) == 0, f"1초 간격 10%↔90% 20초 → 전환 0회 (실제 {len(sw)}회)")

# 창밖을 2초 봤다가 돌아옴
g = SceneGate()
_, sw = feed(g, [(5, 0.05), (2, 0.95), (5, 0.05)])
check(len(sw) == 0 and g.state == "indoor", f"실내에서 2초간 실외 판정 → 전환 없음 (실제 {len(sw)}회)")

# 히스테리시스 구간에 머묾
g = SceneGate()
feed(g, [(1, 0.2)])
_, sw = feed(g, [(20, 0.55)], t0=1.0)
check(len(sw) == 0 and g.state == "indoor" and g.uncertain,
      "실내 상태에서 55%가 20초 유지 → 전환 없음, uncertain (두 모델 병행 신호)")

# 전환 대기 중 진행률
g = SceneGate()
feed(g, [(5, 0.05), (1.6, 0.95)])
prog = g.pending_progress(now=5.0 + 1.6)
check(g.pending and 0.0 < prog < 1.0, f"전환 대기 중 pending=True, 진행률 0~1 (실제 {prog:.2f})")

# 관측 주기에 무관
lats = {}
for hz in (2.0, 5.0, 10.0):
    g = SceneGate()
    _, sw = feed(g, [(5, 0.05), (10, 0.95)], hz=hz)
    lats[hz] = sw[0][0] - 5.0 if sw else float("nan")
spread = max(lats.values()) - min(lats.values())
check(spread <= 0.6, "관측 2/5/10Hz에서 전환 지연 차 0.6초 이내 ("
      + ", ".join(f"{h:g}Hz {v:.1f}s" for h, v in lats.items()) + ")")

# 일시정지: 끊긴 시간이 평활·유지 시간에 섞이지 않음.
# pause() 없이 94초 뒤에 관측하면 '유지 시간 94초'로 간주되어 즉시 전환된다.
g = SceneGate()
feed(g, [(5, 0.05), (1.6, 0.95)])  # 실외 쪽으로 전환 대기 중
p_before = g.p
g.pause()
g.observe(0.95, now=100.0)
check(g.p == p_before, "pause() 뒤 첫 관측은 평활값을 건드리지 않음")
check(g.state == "indoor" and g.pending_progress(now=100.0) == 0.0,
      "pause() 뒤에는 유지 시간을 처음부터 다시 셈 (긴 공백으로 즉시 전환하지 않음)")
g.observe(0.95, now=100.2)
check(g.p is not None and g.p > p_before, "pause() 다음 관측부터 정상 평활 재개")
g2 = SceneGate()
feed(g2, [(5, 0.05), (1.6, 0.95)])
g2.observe(0.95, now=100.0)
check(g2.state == "outdoor", "대조군: pause() 없으면 긴 공백 뒤 즉시 전환됨 (그래서 pause가 필요)")

for bad in ({"enter_indoor": 0.7, "enter_outdoor": 0.6}, {"dwell_s": -1.0},
            {"initial": "cave"}):
    try:
        SceneGate(**bad)
        check(False, f"잘못된 인자 거부: {bad}")
    except ValueError:
        check(True, f"잘못된 인자 거부: {bad}")

if args.with_model:
    print("\n" + "=" * 72)
    print("4. 실제 모델 (GPU)")
    print("=" * 72)
    import time

    import cv2
    import numpy as np

    from tactile_vision.scene import SceneClassifier

    clf = SceneClassifier(CK)
    clf.warmup(10)
    cases = [("demo01.jpg", "outdoor", "도심 거리"), ("demo06.jpg", "outdoor", "철교"),
             ("demo15.jpg", "indoor", "현관 홀·계단"), ("demo13.jpg", "indoor", "전시 진열장")]
    for fname, want, desc in cases:
        img = cv2.imread(str(EXAMPLES / fname))
        if img is None:
            print(f"  SKIP  {fname} 없음")
            continue
        r = clf.classify(cv2.resize(img, (504, 280), interpolation=cv2.INTER_AREA))
        ok = r.p_outdoor > 0.9 if want == "outdoor" else r.p_outdoor < 0.1
        check(ok, f"{fname} ({desc}) → P(실외) {r.p_outdoor * 100:.1f}%, 기대 {want}")
    img = np.zeros((280, 504, 3), np.uint8)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        clf.classify(img)
        ts.append((time.perf_counter() - t0) * 1000)
    med = float(np.median(ts))
    check(med < 20.0, f"판별 1회 중앙값 20ms 미만 (실제 {med:.1f}ms)")

print("\n" + "=" * 72)
print(f"결과: {'전체 통과' if FAIL == 0 else f'{FAIL}개 실패'}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
