#!/usr/bin/env python3
"""
실내/실외 자동 판별 테스트 뷰어 (창 1개).

카메라 원본, 실내 모델 깊이, 실외 모델 깊이를 한 창에 나란히 보여주고,
장면 판별기(Places365)가 지금 화면을 '실외일 확률 몇 %'로 보는지와
그 확률로 자동 전환 규칙(scene.SceneGate)이 어느 모델을 고르는지 표시한다.
사용자가 모드를 바꾸지 않는다 — 전부 자동이다.

창 구성
  위    카메라 | 실내 모델 깊이 | 실외 모델 깊이      (지금 쓰는 모델에 초록 테두리)
  아래  자동 판정 + 확률 막대 | 장면 top-5 | 최근 30초 실외 확률 그래프

자동 전환 규칙 (기본값, 옵션으로 조절)
  1초 평활한 P(실외)가 65%를 넘은 채 1.5초 유지  → 실외 모델
  35% 아래로 1.5초 유지                          → 실내 모델
  그 사이(애매함) 또는 전환 대기 중              → 두 모델 모두, 셀마다 더 가까운 값

키
  q / ESC  종료          space  일시정지          s  스냅샷 저장 (logs/scene_test/)
  c        깊이 색 범위 (8 / 20 / 80 m)          r  판정 초기화
  n / p    다음 / 이전 이미지 (이미지 폴더일 때, 판정도 초기화)

사용
  python tools/scene_test.py                        카메라 (config camera.device)
  python tools/scene_test.py --source 1             카메라 /dev/video1
  python tools/scene_test.py --source walk.mp4      녹화 영상
  python tools/scene_test.py --source photos/       사진 폴더
  python tools/scene_test.py --headless --frames 60 창 없이, 결과 이미지만 저장

비교용이라 매 프레임 두 깊이 모델을 모두 돌린다. 실제 장치는 평소 하나만 돌린다.
창 크기 약 1524×680. 작은 화면은 --scale 0.8.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(TOOLS))

from compare_models import (  # noqa: E402  같은 폴더의 비교 도구와 입력·그리기 규칙을 공유
    INDOOR,
    OUTDOOR,
    KO,
    ModelSpec,
    draw_colorbar,
    draw_grid,
    engine_path_for,
    open_source,
    put,
)
from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision import viz  # noqa: E402
from tactile_vision.camera import Frame  # noqa: E402
from tactile_vision.config import load  # noqa: E402
from tactile_vision.depth import DepthEngine  # noqa: E402
from tactile_vision.scene import SceneClassifier, SceneGate, SceneResult  # noqa: E402

log = logging.getLogger("scene_test")

WIN = "Indoor / Outdoor auto test"
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = 26
GAP = 6
TITLE_H = 36
HEAD_H = 28
INFO_H = 300
FOOT_H = 40
HISTORY_S = 30.0
IN_COL = INDOOR.color  # BGR 하늘색
OUT_COL = OUTDOOR.color  # BGR 주황
USE_COL = (80, 220, 80)
UNC_COL = (0, 215, 255)
DIM = (160, 160, 160)
TEXT = (228, 228, 228)
SEL_TEXT = {"indoor": "INDOOR model", "outdoor": "OUTDOOR model",
            "both": "BOTH models (nearer per cell)"}


@dataclass
class DepthRun:
    spec: ModelSpec
    path: Path
    engine: DepthEngine
    ms: deque = field(default_factory=lambda: deque(maxlen=60))
    cells: np.ndarray | None = None
    depth: np.ndarray | None = None

    def med_ms(self) -> float:
        return float(np.median(self.ms)) if self.ms else 0.0


@dataclass
class HistPoint:
    t: float
    p_now: float  # 이번 판별의 P(실외)
    p_smooth: float  # 게이트의 평활 P(실외)
    state: str
    uncertain: bool


def state_color(state: str | None) -> tuple[int, int, int]:
    if state == "indoor":
        return IN_COL
    if state == "outdoor":
        return OUT_COL
    return (200, 200, 200)


def select(gate: SceneGate, ind: DepthRun, out: DepthRun) -> tuple[str, np.ndarray]:
    """장치가 쓸 셀 거리. 애매하면 두 모델 중 셀마다 더 가까운 값 (틀려도 안전한 쪽)."""
    if gate.uncertain:
        return "both", np.minimum(ind.cells, out.cells)
    return gate.state, (ind.cells if gate.state == "indoor" else out.cells)


# ─────────────────────────────────────────────────────────────
# 그리기 (ASCII만 — cv2.putText는 한글 글리프가 없다)
# ─────────────────────────────────────────────────────────────
def text_w(text: str, scale: float, thick: int = 1) -> int:
    return cv2.getTextSize(text, FONT, scale, thick)[0][0]


def depth_panel(run: DepthRun, rng: tuple[float, float]) -> np.ndarray:
    vis = viz.colorize_depth(run.depth, *rng)
    H, W = vis.shape[:2]
    R, C = geo.GRID_ROWS, geo.GRID_COLS
    ch, cw = H // R, W // C
    for r in range(R):
        for c in range(C):
            x0, y0 = c * cw, r * ch
            cv2.rectangle(vis, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (60, 60, 60), 1)
            put(vis, f"{run.cells[r * C + c]:.1f}m", (x0 + 8, y0 + 20), 0.44,
                (255, 255, 255), outline=True)
    return vis


def panel_header(w: int, text: str, color: tuple[int, int, int],
                 tag: str | None = None, tag_col: tuple[int, int, int] = USE_COL) -> np.ndarray:
    img = np.full((HEAD_H, w, 3), 40, np.uint8)
    put(img, text, (8, 19), 0.5, color)
    if tag:
        tw = text_w(tag, 0.5, 2)
        x = w - tw - 12
        cv2.rectangle(img, (x - 7, 3), (w - 5, HEAD_H - 4), tag_col, -1)
        put(img, tag, (x, 19), 0.5, (15, 15, 15), 2)
    return img


def split_bar(img: np.ndarray, x: int, y: int, w: int, h: int, p_out: float, label: str,
              ticks: tuple[float, ...] = ()) -> None:
    """왼쪽 = 실내 확률(하늘색), 오른쪽 = 실외 확률(주황). ticks = 전환 문턱."""
    p_out = min(1.0, max(0.0, float(p_out)))
    xi = x + int(round((1.0 - p_out) * w))
    cv2.rectangle(img, (x, y), (x + w, y + h), OUT_COL, -1)
    if xi > x:
        cv2.rectangle(img, (x, y), (xi, y + h), IN_COL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (210, 210, 210), 1)
    put(img, f"IN {(1 - p_out) * 100:.0f}%", (x + 7, y + h - 7), 0.52, (15, 15, 15), 2)
    t = f"OUT {p_out * 100:.0f}%"
    put(img, t, (x + w - text_w(t, 0.52, 2) - 7, y + h - 7), 0.52, (15, 15, 15), 2)
    for thr in ticks:
        tx = x + int(round((1.0 - thr) * w))
        cv2.line(img, (tx, y - 5), (tx, y + h + 5), (255, 255, 255), 2)
    put(img, label, (x, y - 8), 0.42, DIM)


def verdict_block(w: int, h: int, gate: SceneGate, res: SceneResult | None, now: float,
                  sel: str, sel_nearest: float) -> np.ndarray:
    img = np.full((h, w, 3), BG, np.uint8)
    put(img, "AUTO DECISION  (no manual switch)", (12, 24), 0.5, DIM)
    if gate.state is None or gate.p is None:
        put(img, "WAITING", (12, 82), 1.7, (200, 200, 200), 4)
        return img

    col = state_color(gate.state)
    big = gate.state.upper()
    put(img, big, (12, 82), 1.7, col, 4)
    pct = gate.p if gate.state == "outdoor" else 1.0 - gate.p
    put(img, f"{pct * 100:.0f}%", (12 + text_w(big, 1.7, 4) + 18, 82), 1.25, col, 3)

    y = 118
    if gate.pending:
        prog = gate.pending_progress(now)
        left = max(0.0, (1.0 - prog) * gate.dwell_s)
        put(img, f"switching to {gate.other.upper()} in {left:.1f} s ...", (12, y), 0.56,
            UNC_COL, 2)
        cv2.rectangle(img, (12, y + 8), (w - 12, y + 15), (70, 70, 70), -1)
        cv2.rectangle(img, (12, y + 8), (12 + int((w - 24) * prog), y + 15), UNC_COL, -1)
    elif gate.uncertain:
        put(img, "not sure -> using BOTH models (nearer)", (12, y), 0.56, UNC_COL, 2)
    else:
        put(img, f"stable -> using {gate.state.upper()} model", (12, y), 0.56, USE_COL, 2)

    if res is not None:
        split_bar(img, 12, 162, w - 24, 26, res.p_outdoor, "this frame (instant)")
    split_bar(img, 12, 218, w - 24, 26, gate.p,
              f"smoothed {gate.tau_s:g} s  -  white ticks = switch at "
              f"{gate.enter_outdoor * 100:.0f}% / {gate.enter_indoor * 100:.0f}%",
              ticks=(gate.enter_outdoor, gate.enter_indoor))

    put(img, f"device output: {SEL_TEXT[sel]}   nearest {sel_nearest:.2f} m", (12, 270),
        0.45, TEXT)
    ago = "-" if gate.last_switch_t is None else f"{now - gate.last_switch_t:.0f} s ago"
    put(img, f"switches {gate.n_switches}   last switch {ago}", (12, 291), 0.45, DIM)
    return img


def scenes_block(w: int, h: int, res: SceneResult | None, clf_ms: float,
                 scene_hz: float) -> np.ndarray:
    img = np.full((h, w, 3), BG, np.uint8)
    put(img, "TOP-5 SCENES  (what the scene model sees)", (12, 24), 0.5, DIM)
    if res is None:
        put(img, "waiting for first result ...", (12, 62), 0.5, DIM)
        return img
    bar_w = w - 100
    for k, (name, p, is_out) in enumerate(res.top):
        y = 60 + k * 42
        col = OUT_COL if is_out else IN_COL
        tag = "OUT" if is_out else "IN"
        cv2.rectangle(img, (12, y - 16), (58, y + 4), col, -1)
        put(img, tag, (35 - text_w(tag, 0.45, 2) // 2, y - 1), 0.45, (15, 15, 15), 2)
        put(img, name.replace("_", " ")[:32], (68, y), 0.53, TEXT)
        cv2.rectangle(img, (12, y + 9), (12 + bar_w, y + 17), (55, 55, 55), -1)
        cv2.rectangle(img, (12, y + 9), (12 + int(bar_w * p), y + 17), col, -1)
        t = f"{p * 100:.1f}%"
        put(img, t, (w - text_w(t, 0.5) - 12, y + 17), 0.5, TEXT)
    put(img, "P(outdoor) = sum over all 204 outdoor scene types", (12, h - 34), 0.42, DIM)
    put(img, f"scene model {clf_ms:.1f} ms, {scene_hz:g} times per second", (12, h - 12),
        0.42, DIM)
    return img


def trend_block(w: int, h: int, hist: deque, now: float, gate: SceneGate) -> np.ndarray:
    img = np.full((h, w, 3), BG, np.uint8)
    put(img, f"P(outdoor)  last {HISTORY_S:.0f} s", (12, 24), 0.5, DIM)
    legend = "dots instant  line smoothed  band decision"
    put(img, legend, (w - text_w(legend, 0.38) - 12, 24), 0.38, DIM)
    gx0, gy0, gx1, gy1 = 46, 40, w - 14, h - 34
    span, hgt = gx1 - gx0, gy1 - gy0
    cv2.rectangle(img, (gx0, gy0), (gx1, gy1), (38, 38, 38), -1)

    def X(t: float) -> int:
        return int(round(gx1 - span * (now - t) / HISTORY_S))

    def Y(p: float) -> int:
        return int(round(gy1 - hgt * min(1.0, max(0.0, p))))

    pts = [e for e in hist if now - e.t <= HISTORY_S + 1.0]
    # 판정 띠: 관측 i의 판정이 다음 관측까지 유지된 구간
    for i, e in enumerate(pts):
        xa = max(gx0, X(e.t))
        xb = min(gx1, X(pts[i + 1].t) if i + 1 < len(pts) else X(now))
        if xb <= xa:
            continue
        tint = tuple(int(c * 0.3) for c in state_color(e.state))
        cv2.rectangle(img, (xa, gy0), (xb, gy1), tint, -1)
        if e.uncertain:
            cv2.rectangle(img, (xa, gy1 - 7), (xb, gy1), UNC_COL, -1)

    for thr in (gate.enter_outdoor, gate.enter_indoor):
        yy = Y(thr)
        for xx in range(gx0, gx1, 10):
            cv2.line(img, (xx, yy), (min(xx + 5, gx1), yy), (175, 175, 175), 1)
        put(img, f"{thr * 100:.0f}", (gx0 - 26, yy + 4), 0.38, (175, 175, 175))
    put(img, "OUT", (6, gy0 + 12), 0.42, OUT_COL)
    put(img, "IN", (10, gy1), 0.42, IN_COL)

    line = []
    for e in pts:
        if now - e.t > HISTORY_S:
            continue
        x = X(e.t)
        cv2.circle(img, (x, Y(e.p_now)), 2, (185, 185, 185), -1, cv2.LINE_AA)
        line.append((x, Y(e.p_smooth)))
    if len(line) >= 2:
        cv2.polylines(img, [np.array(line, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)

    for s in (30, 20, 10, 0):
        x = gx1 - int(round(span * s / HISTORY_S))
        lab = "now" if s == 0 else f"-{s}s"
        put(img, lab, (min(max(x - text_w(lab, 0.4) // 2, 4), w - text_w(lab, 0.4) - 4),
                       gy1 + 18), 0.4, DIM)
    cv2.rectangle(img, (gx0, gy0), (gx1, gy1), (90, 90, 90), 1)
    return img


def compose(frame: Frame, ind: DepthRun, out: DepthRun, gate: SceneGate,
            res: SceneResult | None, hist: deque, now: float, rng: tuple[float, float],
            clf_ms: float, scene_hz: float, source_label: str, loop_fps: float,
            paused: bool) -> np.ndarray:
    pw, ph = frame.image.shape[1], frame.image.shape[0]
    W = 3 * pw + 2 * GAP
    sel, sel_cells = select(gate, ind, out)

    # 제목
    title = np.full((TITLE_H, W, 3), 40, np.uint8)
    put(title, "INDOOR / OUTDOOR AUTO-SWITCH TEST", (12, 25), 0.7, (240, 240, 240), 2)
    right = f"{source_label}   frame #{frame.index}   loop {loop_fps:4.1f} fps"
    if paused:
        right += "   PAUSED"
    put(title, right, (W - text_w(right, 0.5) - 12, 24), 0.5,
        (80, 220, 255) if paused else (200, 200, 200))

    # 위: 카메라 | 실내 | 실외
    cam = frame.image.copy()
    draw_grid(cam)
    p_in, p_out = depth_panel(ind, rng), depth_panel(out, rng)
    tag_in = tag_out = None
    tag_col = USE_COL
    if sel == "both":
        tag_in = tag_out = "BOTH"
        tag_col = UNC_COL
    elif sel == "indoor":
        tag_in = "IN USE"
    else:
        tag_out = "IN USE"
    for panel, tag in ((p_in, tag_in), (p_out, tag_out)):
        if tag:
            cv2.rectangle(panel, (2, 2), (pw - 3, ph - 3), tag_col, 4)
    h_cam = panel_header(pw, "CAMERA  (same frame goes to both models)", TEXT)
    h_in = panel_header(pw, f"INDOOR model   nearest {ind.cells.min():.2f} m   "
                            f"{ind.med_ms():.0f} ms", IN_COL, tag_in, tag_col)
    h_out = panel_header(pw, f"OUTDOOR model   nearest {out.cells.min():.2f} m   "
                             f"{out.med_ms():.0f} ms", OUT_COL, tag_out, tag_col)
    gcol = np.full((HEAD_H + ph, GAP, 3), BG, np.uint8)
    top = np.hstack([np.vstack([h_cam, cam]), gcol, np.vstack([h_in, p_in]), gcol,
                     np.vstack([h_out, p_out])])

    # 아래: 판정 | 장면 | 추이
    gi = np.full((INFO_H, GAP, 3), BG, np.uint8)
    info = np.hstack([
        verdict_block(pw, INFO_H, gate, res, now, sel, float(sel_cells.min())), gi,
        scenes_block(pw, INFO_H, res, clf_ms, scene_hz), gi,
        trend_block(pw, INFO_H, hist, now, gate),
    ])

    foot = np.full((FOOT_H, W, 3), BG, np.uint8)
    put(foot, "depth colour", (12, 15), 0.4, DIM)
    draw_colorbar(foot, 110, 4, 380, rng)
    keys = "q quit   space pause   s save   c colour   r reset decision   n/p image"
    put(foot, keys, (W - text_w(keys, 0.44) - 12, 24), 0.44, (140, 140, 140))

    sep = np.full((GAP, W, 3), BG, np.uint8)
    return np.vstack([title, top, sep, info, foot])


# ─────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────
def save_snapshot(out_dir: Path, canvas: np.ndarray, frame: Frame, gate: SceneGate,
                  res: SceneResult | None, sel: str, source_label: str) -> Path:
    """창 이미지 + 입력 + scene_log.csv 한 줄. true_label은 나중에 실제 장소를 적는 칸."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    base = out_dir / stamp
    cv2.imwrite(f"{base}_view.png", canvas)
    cv2.imwrite(f"{base}_input.png", frame.image)
    csv_path = out_dir / "scene_log.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["stamp", "source", "frame", "p_outdoor_instant", "p_outdoor_smoothed",
                         "decision", "uncertain", "device_output", "top1_scene", "top1_prob",
                         "top1_is_outdoor", "true_label"])
        top1 = res.top[0] if res is not None else ("", 0.0, False)
        wr.writerow([stamp, source_label, frame.index,
                     f"{res.p_outdoor:.4f}" if res is not None else "",
                     f"{gate.p:.4f}" if gate.p is not None else "",
                     gate.state or "", int(gate.uncertain), sel,
                     top1[0], f"{top1[1]:.4f}", int(top1[2]), ""])
    log.info("스냅샷 저장: %s_{view,input}.png", base)
    return base


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="설정 파일 (기본 config/default.yaml)")
    ap.add_argument("--source", default=None,
                    help="카메라 번호 | 영상 | 이미지 | 이미지 폴더 (기본: 설정의 카메라)")
    ap.add_argument("--scene-hz", type=float, default=5.0, help="장면 판별 횟수/초 (기본 5)")
    ap.add_argument("--tau", type=float, default=1.0, help="확률 평활 시간 초 (기본 1.0)")
    ap.add_argument("--enter-outdoor", type=float, default=0.65,
                    help="실외로 바꾸는 평활 확률 (기본 0.65)")
    ap.add_argument("--enter-indoor", type=float, default=0.35,
                    help="실내로 바꾸는 평활 확률 (기본 0.35)")
    ap.add_argument("--dwell", type=float, default=1.5, help="전환 조건 유지 시간 초 (기본 1.5)")
    ap.add_argument("--headless", action="store_true", help="창 없이 실행, 결과 이미지 저장")
    ap.add_argument("--frames", type=int, default=0,
                    help="처리할 프레임 수 (0=무한, headless 기본 60). "
                         "headless + 이미지 폴더면 이미지당 프레임 수")
    ap.add_argument("--duration", type=float, default=0.0, help="지정 초 후 자동 종료 (0=무한)")
    ap.add_argument("--scale", type=float, default=1.0, help="창 크기 배율")
    ap.add_argument("--pos", default="40,50", metavar="X,Y", help="창 왼쪽 위 위치 (기본 40,50)")
    ap.add_argument("--out-dir", default="logs/scene_test", help="스냅샷 저장 폴더")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if args.scene_hz <= 0:
        log.error("--scene-hz는 0보다 커야 함")
        return 1
    try:
        win_x, win_y = (int(v) for v in args.pos.split(","))
    except ValueError:
        log.error("--pos 형식은 X,Y (예: 40,50): %s", args.pos)
        return 1
    gate_kw = dict(tau_s=args.tau, enter_outdoor=args.enter_outdoor,
                   enter_indoor=args.enter_indoor, dwell_s=args.dwell)
    try:
        gate = SceneGate(**gate_kw)
    except ValueError as e:
        log.error("판정 설정 오류: %s", e)
        return 1

    cfg = load(args.config)
    cfg.validate()
    spec = cfg.depth.spec

    # Qt는 디스플레이가 없으면 예외 없이 abort한다 → GUI 호출 전에 확인
    if not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log.warning("DISPLAY 없음 — headless로 전환 (결과는 %s)", args.out_dir)
        args.headless = True
    if args.headless and args.frames <= 0 and args.duration <= 0:
        args.frames = 60

    runs: list[DepthRun] = []
    for ms_, flag in ((INDOOR, ""), (OUTDOOR, " --outdoor")):
        p = engine_path_for(ms_, None, cfg, spec)
        if not p.exists():
            log.error("%s 엔진 없음: %s\n  빌드: python tools/build_engine.py --spec %s%s",
                      KO[ms_.key], p, spec.name, flag)
            return 1
        dcfg = dataclasses.replace(cfg.depth, engine=str(p), max_depth_m=ms_.max_depth_m)
        runs.append(DepthRun(ms_, p, DepthEngine(dcfg, cfg.grid, p)))
    try:
        clf = SceneClassifier(cfg.path("checkpoints/places365"))
    except (FileNotFoundError, RuntimeError) as e:
        log.error("장면 판별기 로드 실패: %s", e)
        return 1
    for r in runs:
        r.engine.warmup(10)
    clf.warmup(10)
    ind, out = runs

    try:
        source, kind, source_label = open_source(args.source, cfg, spec)
    except Exception as e:  # noqa: BLE001
        log.error("소스 열기 실패: %s", e)
        return 1

    out_dir = cfg.path(args.out_dir)
    rngs = [(cfg.mapping.near_m, cfg.mapping.far_m), (0.5, 20.0), (0.5, 80.0)]
    range_idx = 0
    period = 1.0 / args.scene_hz
    hist: deque = deque()
    clf_ms: deque = deque(maxlen=60)
    res: SceneResult | None = None
    t_scene: float | None = None
    frame: Frame | None = None
    canvas: np.ndarray | None = None
    paused = False
    t_pause = 0.0
    need_frame = False
    step_image = 0
    n_proc = per_image = n_scene = 0
    saved_at = -1
    misses = 0
    placed = False
    loop_dt: deque = deque(maxlen=30)
    t_prev: float | None = None
    state_counts = {"indoor": 0, "outdoor": 0, "uncertain": 0}

    def label_now() -> str:
        return source.label if kind == "images" else source_label

    def render(now: float) -> np.ndarray:
        fps = 1.0 / float(np.mean(loop_dt)) if loop_dt else 0.0
        med = float(np.median(clf_ms)) if clf_ms else 0.0
        return compose(frame, ind, out, gate, res, hist, now, rngs[range_idx], med,
                       args.scene_hz, label_now(), fps, paused)

    def reset_decision() -> None:
        nonlocal gate, res, t_scene
        gate = SceneGate(**gate_kw)
        hist.clear()
        res = None
        t_scene = None

    if not args.headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        log.info("시작 — 키: q 종료 / space 일시정지 / s 저장 / c 색범위 / r 판정 초기화")
    else:
        log.info("시작 — headless, 결과는 %s", out_dir)
    t_start = time.perf_counter()

    try:
        while True:
            if args.duration > 0 and time.perf_counter() - t_start > args.duration:
                log.info("지정 시간 경과 — 종료")
                break

            if step_image and kind == "images":
                source.step(step_image)
                step_image = 0
                need_frame = True
                reset_decision()

            if frame is None or not paused or need_frame:
                f = source.read(timeout=1.0)
                if f is None:
                    if kind != "camera":
                        log.info("소스 끝")
                        break
                    misses += 1
                    if misses % 5 == 1:
                        log.warning("카메라 프레임 없음 (%d회 연속)", misses)
                    if placed:
                        cv2.waitKey(1)
                    continue
                misses = 0
                frame = f
                tp = time.perf_counter()
                if t_prev is not None:
                    loop_dt.append(tp - t_prev)
                t_prev = tp

                # 같은 프레임을 두 깊이 모델에 (순서를 번갈아 시간 편향 제거)
                for r in (runs if n_proc % 2 == 0 else runs[::-1]):
                    t0 = time.perf_counter()
                    r.cells = r.engine.infer(frame.image)
                    r.ms.append((time.perf_counter() - t0) * 1000)
                for r in runs:
                    r.depth = r.engine.last_depth_map()

                # 장면 판별은 초당 scene_hz 회 (장면은 천천히 바뀐다)
                now = time.monotonic()
                if t_scene is None or now - t_scene >= period or need_frame:
                    res = clf.classify(frame.image)
                    clf_ms.append(res.ms)
                    gate.observe(res.p_outdoor, now=now)
                    t_scene = now
                    n_scene += 1
                    hist.append(HistPoint(now, res.p_outdoor, gate.p, gate.state,
                                          gate.uncertain))
                    while hist and now - hist[0].t > HISTORY_S + 2.0:
                        hist.popleft()
                    state_counts["uncertain" if gate.uncertain else gate.state] += 1
                n_proc += 1
                per_image += 1
                need_frame = False

            now = t_pause if paused else time.monotonic()

            # ── headless: 저장할 때만 그린다 ──
            if args.headless:
                done_image = kind == "images" and args.frames > 0 and per_image >= args.frames
                done_all = kind != "images" and args.frames > 0 and n_proc >= args.frames
                if done_image or done_all:
                    canvas = render(now)
                    sel, _ = select(gate, ind, out)
                    save_snapshot(out_dir, canvas, frame, gate, res, sel, label_now())
                    log.info("판정: %s → %s (평활 P(실외) %.0f%%, 이번 %.0f%%, top1 %s)",
                             label_now(), "uncertain/" + str(gate.state) if gate.uncertain
                             else gate.state, (gate.p or 0) * 100,
                             (res.p_outdoor if res else 0) * 100,
                             res.top[0][0] if res else "-")
                    saved_at = n_proc
                    if done_all or source.i + 1 >= source.count:
                        break
                    step_image = 1
                    per_image = 0
                continue

            # ── 창 표시 ──
            canvas = render(now)
            cv2.imshow(WIN, canvas)
            if not placed:
                ch_, cw_ = canvas.shape[:2]
                cv2.resizeWindow(WIN, int(cw_ * args.scale), int(ch_ * args.scale))
                cv2.moveWindow(WIN, win_x, win_y)
                placed = True

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                if paused:
                    t_pause = time.monotonic()
                    gate.pause()  # 멈춘 시간이 평활·유지 시간에 섞이지 않게
                else:
                    t_scene = None  # 재개 즉시 판별
                    t_prev = None
                    loop_dt.clear()
            elif key == ord("s") and frame is not None:
                sel, _ = select(gate, ind, out)
                save_snapshot(out_dir, canvas, frame, gate, res, sel, label_now())
            elif key == ord("c"):
                range_idx = (range_idx + 1) % len(rngs)
            elif key == ord("r"):
                reset_decision()
                need_frame = True
                log.info("판정 초기화")
            elif key in (ord("n"), ord("p")) and kind == "images":
                step_image = 1 if key == ord("n") else -1

            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break  # 창 닫기 버튼
            if args.frames > 0 and n_proc >= args.frames:
                break
    except KeyboardInterrupt:
        log.info("Ctrl+C — 종료")
    finally:
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()

    if args.headless and frame is not None and saved_at != n_proc:
        canvas = render(t_pause if paused else time.monotonic())
        sel, _ = select(gate, ind, out)
        save_snapshot(out_dir, canvas, frame, gate, res, sel, label_now())

    el = time.perf_counter() - t_start
    total = max(1, sum(state_counts.values()))
    print()
    print(f"── 실내/실외 자동 판별 요약: {source_label} / {n_proc} 프레임 / {el:.1f}초 ──")
    print(f"  깊이 추론 (중앙값)  실내 {ind.med_ms():.1f} ms / 실외 {out.med_ms():.1f} ms")
    if clf_ms:
        print(f"  장면 판별  {np.median(clf_ms):.1f} ms × {n_scene}회")
    print(f"  판정 비율  실내 {state_counts['indoor'] / total * 100:.0f}%  "
          f"실외 {state_counts['outdoor'] / total * 100:.0f}%  "
          f"애매(두 모델) {state_counts['uncertain'] / total * 100:.0f}%")
    if gate.p is not None:
        print(f"  마지막 판정 {gate.state}  (평활 P(실외) {gate.p * 100:.0f}%)  "
              f"전환 {gate.n_switches}회")
    return 0


if __name__ == "__main__":
    sys.exit(main())
