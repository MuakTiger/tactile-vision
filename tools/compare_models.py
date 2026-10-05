#!/usr/bin/env python3
"""
실내(Hypersim) / 실외(Virtual KITTI 2) metric 깊이 모델 비교 뷰어.

같은 프레임을 두 TRT 엔진에 넣고, 모델마다 창을 1개씩 나란히 띄운다.
색 범위·셀 풀링·로그 매핑·비대칭 필터는 두 창이 완전히 같으므로
화면에 보이는 차이는 전부 모델에서 나온다.

창 구성 (모델마다 1개)
  제목   모델 이름 + 프레임 번호 (두 창 번호가 같으면 같은 프레임)
  상단   모델 입력(격자) | 깊이맵(고정 색 범위) + 셀 거리/레벨
  하단   그 모델로 구동했을 때의 촉각 패드 | 추론 속도, 두 모델 비교 수치, 색 막대

키 (둘 중 아무 창이나 포커스된 상태에서)
  q / ESC   종료                     space   일시정지 / 재개
  s         스냅샷 저장 (logs/compare/ — 창 이미지, 입력, 깊이 npz, probes.csv)
  c         색 범위 전환: 매핑 범위(0.5~8m) → 0.5~20m → 0.5~80m
  n / p     다음 / 이전 이미지 (이미지 폴더 입력일 때)
  왼쪽 클릭   그 지점에서 두 모델의 거리 표시 (줄자 비교용)    오른쪽 클릭  해제

사용
  python tools/compare_models.py                         카메라 (config camera.device)
  python tools/compare_models.py --source 1              카메라 /dev/video1
  python tools/compare_models.py --source walk.mp4       녹화 영상 (반복 재생)
  python tools/compare_models.py --source photo.jpg      정지 이미지
  python tools/compare_models.py --source photos/        이미지 폴더
  python tools/compare_models.py --headless --frames 30  창 없이 실행, 결과만 저장
  python tools/compare_models.py --probe 252,140         처음부터 화면 중앙 거리 비교

엔진이 없으면
  python tools/build_engine.py --spec wide             실내
  python tools/build_engine.py --spec wide --outdoor   실외

주의: 비교용 도구라 매 프레임 두 모델을 모두 돌린다. 여기서 나오는 루프 fps는
실제 장치(모델 1개 + 위험도 검출)의 속도가 아니다. 모델별 추론 시간을 볼 것.
시간 필터는 프레임 단위라, 루프가 느린 이 도구에서는 패드 반응도 실제 장치보다 느리다.
두 창 합계 폭은 약 2050px. 1920 폭 화면(원격 접속 등)은 --scale 0.9 를 쓴다.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision import viz  # noqa: E402
from tactile_vision.camera import Camera, Frame, VideoFileSource, crop_rect  # noqa: E402
from tactile_vision.config import Config, load  # noqa: E402
from tactile_vision.depth import DepthEngine  # noqa: E402
from tactile_vision.mapping import DepthMapper  # noqa: E402

log = logging.getLogger("compare")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = 26
TITLE_H = 30
PAD_CELL_PX = 64
PROBE_HALF = 2  # 클릭 지점 주변 5x5 픽셀의 중앙값을 읽는다 (단일 픽셀 노이즈 회피)
WINDOW_GAP = 12
# GPU가 잠시 쉬면 클럭이 내려가 다음 첫 추론이 느려진다(DVFS). 실측(실외 엔진):
#   유휴 0.10s → 11.8ms,  0.20s → 21.2ms,  0.25s 이상 → 약 30ms  (연속 실행 11.4ms)
# 저장·일시정지 직후 같은 프레임은 결과는 쓰되 추론 시간 통계에서는 뺀다.
# 모델 속도가 아니라 클럭 상태를 재는 값이기 때문. 정상 루프 간격(~40ms)은 해당 없음.
IDLE_GAP_S = 0.15
KO = {"indoor": "실내", "outdoor": "실외"}


# ─────────────────────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    key: str  # indoor | outdoor
    label: str  # 창 제목. cv2.putText는 한글을 못 그리므로 ASCII
    tag: str  # tools/build_engine.py 가 쓰는 파일명 태그
    max_depth_m: float  # 학습 시 max_depth. 엔진에 이미 포함됨 — 표시용
    color: tuple[int, int, int]  # BGR 강조색


INDOOR = ModelSpec("indoor", "INDOOR - Hypersim", "hypersim", 20.0, (255, 190, 60))
OUTDOOR = ModelSpec("outdoor", "OUTDOOR - Virtual KITTI 2", "vkitti", 80.0, (60, 170, 255))


@dataclass
class ModelRun:
    """모델 1개의 엔진·필터 상태와 직전 프레임 결과."""

    spec: ModelSpec
    engine_path: Path
    engine: DepthEngine
    mapper: DepthMapper
    recent_ms: deque = field(default_factory=lambda: deque(maxlen=120))
    all_ms: list = field(default_factory=list)
    cells: np.ndarray | None = None  # (24,) 셀 대표 거리 m
    levels: np.ndarray | None = None  # (24,) 필터 후 정수 레벨
    depth: np.ndarray | None = None  # (H, W) 전체 깊이맵 m

    @property
    def win(self) -> str:
        return self.spec.label


@dataclass
class ViewState:
    paused: bool = False
    probe: tuple[int, int] | None = None  # 모델 입력 좌표 (x, y)
    range_idx: int = 0
    step_image: int = 0  # n/p 요청 (+1 / -1)
    need_frame: bool = True  # 일시정지 중에도 새 프레임 1장 처리 요청


# ─────────────────────────────────────────────────────────────
# 입력 소스
# ─────────────────────────────────────────────────────────────
class ImageSource:
    """
    정지 이미지 1장 또는 폴더. read()마다 현재 이미지를 새 Frame으로 돌려준다.
    카메라 경로와 똑같이 종횡비 센터 크롭 후 INTER_AREA 축소한다.
    """

    def __init__(self, paths: list[Path], out_w: int, out_h: int) -> None:
        self.paths = paths
        self.out_w, self.out_h = out_w, out_h
        self.i = 0
        self.n = 0
        self._load()

    def _load(self) -> None:
        p = self.paths[self.i]
        raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError(f"이미지를 읽을 수 없음: {p}")
        x, y, w, h = crop_rect(raw.shape[1], raw.shape[0], self.out_w / self.out_h)
        self._raw = raw
        self._img = cv2.resize(
            raw[y : y + h, x : x + w], (self.out_w, self.out_h), interpolation=cv2.INTER_AREA
        )
        log.info("이미지 %d/%d: %s (%dx%d)", self.i + 1, len(self.paths), p.name,
                 raw.shape[1], raw.shape[0])

    @property
    def label(self) -> str:
        return f"{self.paths[self.i].name} ({self.i + 1}/{len(self.paths)})"

    @property
    def count(self) -> int:
        return len(self.paths)

    def step(self, d: int) -> None:
        self.i = (self.i + d) % len(self.paths)
        self._load()

    def read(self, timeout: float = 1.0) -> Frame:
        self.n += 1
        return Frame(image=self._img, raw=self._raw, ts=time.perf_counter(), index=self.n)

    def close(self) -> None:
        pass


def open_source(arg: str | None, cfg: Config, spec: geo.InputSpec):
    """(source, kind, label) — kind: camera | video | images"""
    w, h = spec.width, spec.height
    if arg is None or arg.isdigit():
        cam_cfg = cfg.camera if arg is None else dataclasses.replace(cfg.camera, device=int(arg))
        cam = Camera(cam_cfg, w, h)
        cam.start()
        return cam, "camera", f"camera {cam_cfg.device}"

    p = Path(arg).expanduser()
    if p.is_dir():
        imgs = sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        if not imgs:
            raise FileNotFoundError(f"폴더에 이미지가 없음: {p}")
        return ImageSource(imgs, w, h), "images", str(p)
    if not p.exists():
        raise FileNotFoundError(f"소스 없음: {p}")
    if p.suffix.lower() in IMAGE_EXTS:
        return ImageSource([p], w, h), "images", str(p)
    return VideoFileSource(str(p), w, h, loop=True), "video", p.name


# ─────────────────────────────────────────────────────────────
# 수치
# ─────────────────────────────────────────────────────────────
def probe_value(depth: np.ndarray, p: tuple[int, int]) -> float:
    x, y = p
    h, w = depth.shape
    patch = depth[max(0, y - PROBE_HALF) : min(h, y + PROBE_HALF + 1),
                  max(0, x - PROBE_HALF) : min(w, x + PROBE_HALF + 1)]
    return float(np.median(patch))


def cross_stats(ind: ModelRun, out: ModelRun) -> dict[str, float]:
    """두 모델의 셀 단위 비교. 비율은 항상 실외/실내로 통일한다."""
    ratio = out.cells / np.maximum(ind.cells, 1e-3)
    dl = out.levels.astype(np.int32) - ind.levels.astype(np.int32)  # +면 실외가 더 가깝다고 봄
    return {
        "ratio_med": float(np.median(ratio)),
        "ratio_lo": float(ratio.min()),
        "ratio_hi": float(ratio.max()),
        "ndiff": int(np.count_nonzero(dl)),
        "mean_abs_dl": float(np.abs(dl).mean()),
        "out_nearer": int(np.count_nonzero(dl > 0)),
        "in_nearer": int(np.count_nonzero(dl < 0)),
    }


def ticks_for(near: float, far: float) -> list[float]:
    """선형 색 막대용 눈금. 5개 안팎."""
    step = 50.0
    for s in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        if (far - near) / s <= 5:
            step = s
            break
    rest = [float(v) for v in np.arange(step, far + 1e-6, step) if v > near + 0.4 * step]
    return [near] + rest


# ─────────────────────────────────────────────────────────────
# 그리기 (ASCII만 — cv2.putText는 한글 글리프가 없다)
# ─────────────────────────────────────────────────────────────
def put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.42,
        color: tuple[int, int, int] = (215, 215, 215), thick: int = 1,
        outline: bool = False) -> None:
    if outline:  # 컬러맵 위 글자 가독성
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_grid(img: np.ndarray) -> None:
    H, W = img.shape[:2]
    R, C = geo.GRID_ROWS, geo.GRID_COLS
    for c in range(1, C):
        x = c * W // C
        cv2.line(img, (x, 0), (x, H - 1), (90, 90, 90), 1)
    for r in range(1, R):
        y = r * H // R
        cv2.line(img, (0, y), (W - 1, y), (90, 90, 90), 1)
    xm = (C // 2) * W // C  # 좌우 손 경계
    cv2.line(img, (xm, 0), (xm, H - 1), (0, 200, 200), 1)


def draw_probe(img: np.ndarray, p: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = p
    cv2.drawMarker(img, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 24, 4, cv2.LINE_AA)
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x - PROBE_HALF - 1, y - PROBE_HALF - 1),
                  (x + PROBE_HALF + 1, y + PROBE_HALF + 1), (0, 0, 0), 1)


def depth_panel(run: ModelRun, cfg: Config, rng: tuple[float, float],
                probe: tuple[int, int] | None) -> np.ndarray:
    vis = viz.colorize_depth(run.depth, *rng)
    H, W = vis.shape[:2]
    R, C = geo.GRID_ROWS, geo.GRID_COLS
    ch, cw = H // R, W // C
    for r in range(R):
        for c in range(C):
            i = r * C + c
            x0, y0 = c * cw, r * ch
            cv2.rectangle(vis, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (60, 60, 60), 1)
            put(vis, f"{run.cells[i]:.1f}m", (x0 + 4, y0 + 15), 0.40, (255, 255, 255),
                outline=True)
            lv = int(run.levels[i])
            put(vis, f"L{lv}", (x0 + 4, y0 + ch - 7), 0.40,
                viz.level_color(lv, cfg.mapping.levels), outline=True)
    if probe is not None:
        draw_probe(vis, probe, (255, 255, 255))
    return vis


def draw_colorbar(img: np.ndarray, x: int, y: int, width: int,
                  rng: tuple[float, float]) -> None:
    near, far = rng
    ramp = np.repeat(np.linspace(near, far, width, dtype=np.float32)[None, :], 12, axis=0)
    img[y : y + 12, x : x + width] = viz.colorize_depth(ramp, near, far)
    for t in ticks_for(near, far):
        tx = x + int(round((t - near) / (far - near) * (width - 1)))
        cv2.line(img, (tx, y + 12), (tx, y + 16), (200, 200, 200), 1)
        lab = f"{t:g}"
        (tw, _), _ = cv2.getTextSize(lab, FONT, 0.36, 1)
        put(img, lab, (max(x, min(tx - tw // 2, x + width - tw)), y + 29), 0.36)
    put(img, "m", (x + width + 6, y + 11), 0.38)


def info_panel(run: ModelRun, ind: ModelRun, out: ModelRun, cfg: Config, st: ViewState,
               rng: tuple[float, float], xs: dict[str, float], w: int, h: int,
               source_label: str, loop_fps: float) -> np.ndarray:
    img = np.full((h, w, 3), BG, np.uint8)
    dim, white, cross = (160, 160, 160), (225, 225, 225), (230, 210, 140)
    if run.recent_ms:
        a = np.asarray(run.recent_ms)
        infer_line = (f"infer   avg {a.mean():5.1f} ms   p95 {np.percentile(a, 95):5.1f} ms"
                      f"   (last {len(run.recent_ms)})")
    else:
        infer_line = "infer   -- (first frame after idle is not timed)"
    m = cfg.mapping
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"engine  {run.engine_path.name}", dim),
        (infer_line, white),
        (f"loop    {loop_fps:4.1f} fps (both models + display)   "
         f"{'PAUSED' if st.paused else 'LIVE'}", (80, 220, 255) if st.paused else white),
        (f"source  {source_label if len(source_label) <= 60 else '...' + source_label[-57:]}",
         dim),
        (f"cells   nearest {run.cells.min():5.2f} m   median {np.median(run.cells):5.2f} m"
         f"   farthest {run.cells.max():5.1f} m", white),
        ("", white),
        (f"OUT/IN cell distance ratio  median x{xs['ratio_med']:.2f}"
         f"   [x{xs['ratio_lo']:.2f} - x{xs['ratio_hi']:.2f}]", cross),
        (f"levels differ {xs['ndiff']:2d}/24   mean|dL| {xs['mean_abs_dl']:.2f}"
         f"   OUT nearer {xs['out_nearer']}  IN nearer {xs['in_nearer']}", cross),
    ]
    if st.probe is not None:
        v_in, v_out = probe_value(ind.depth, st.probe), probe_value(out.depth, st.probe)
        lines.append((f"probe ({st.probe[0]},{st.probe[1]})   IN {v_in:5.2f} m   "
                      f"OUT {v_out:5.2f} m   OUT/IN x{v_out / max(v_in, 1e-3):.2f}",
                      (120, 255, 160)))
    else:
        lines.append(("probe   left-click a point to read both models there", dim))
    lines.append((f"mapping {m.mode} {m.near_m:g}-{m.far_m:g} m, {m.levels} levels"
                  f"      colour {rng[0]:g}-{rng[1]:g} m  [c]", dim))

    y = 22
    for text, col in lines:
        if text:
            put(img, text, (10, y), 0.42, col)
        y += 20
    draw_colorbar(img, 12, y - 6, min(420, w - 50), rng)
    put(img, "q quit  space pause  s save  c colour  n/p image  L-click probe  R-click clear",
        (10, h - 10), 0.36, (130, 130, 130))
    return img


def title_strip(run: ModelRun, w: int, frame_label: str) -> np.ndarray:
    img = np.full((TITLE_H, w, 3), 40, np.uint8)
    put(img, f"{run.spec.label}   (trained max_depth {run.spec.max_depth_m:g} m)",
        (10, 21), 0.6, run.spec.color, 2)
    (tw, _), _ = cv2.getTextSize(frame_label, FONT, 0.45, 1)
    put(img, frame_label, (w - tw - 10, 20), 0.45, (200, 200, 200))
    return img


def compose(run: ModelRun, ind: ModelRun, out: ModelRun, frame: Frame, cfg: Config,
            st: ViewState, rng: tuple[float, float], xs: dict[str, float],
            source_label: str, loop_fps: float) -> np.ndarray:
    cam = frame.image.copy()
    draw_grid(cam)
    if st.probe is not None:
        draw_probe(cam, st.probe, run.spec.color)
    top = np.hstack([cam, depth_panel(run, cfg, rng, st.probe)])
    pad = viz.draw_tactile_pad(run.levels, np.zeros(geo.CELL_COUNT, np.int16), cfg,
                               cell_px=PAD_CELL_PX)
    info = info_panel(run, ind, out, cfg, st, rng, xs, top.shape[1] - pad.shape[1],
                      pad.shape[0], source_label, loop_fps)
    title = title_strip(run, top.shape[1], f"frame #{frame.index}")
    return np.vstack([title, top, np.hstack([pad, info])])


# ─────────────────────────────────────────────────────────────
# 창 배치
# ─────────────────────────────────────────────────────────────
def place_windows(runs: list[ModelRun], ww: int, wh: int) -> None:
    """
    두 창을 겹치지 않게 나란히 놓는다.

    창 관리자는 도크·상단바를 피하려고 요청 위치를 바꾼다(GNOME은 왼쪽 도크 폭만큼
    오른쪽으로 민다). 계산한 좌표로 두 번째 창을 놓으면 그만큼 겹치므로, 앞 창이
    실제로 놓인 위치를 읽어 다음 창의 x를 정한다. Qt는 move 직후엔 요청 좌표를
    그대로 돌려주므로, 창 관리자가 위치를 확정할 시간을 먼저 준다.
    """
    x = 20
    for r in runs:
        cv2.resizeWindow(r.win, ww, wh)
        cv2.moveWindow(r.win, x, 40)
        t_end = time.perf_counter() + 0.3
        while time.perf_counter() < t_end:
            cv2.waitKey(10)
        rx, _, rw, _ = cv2.getWindowImageRect(r.win)
        right = rx + rw if rw > 0 else x + ww
        x = max(x + ww, right) + WINDOW_GAP


# ─────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────
def save_snapshot(out_dir: Path, runs: list[ModelRun], canvases: list[np.ndarray],
                  frame: Frame, st: ViewState, source_label: str,
                  xs: dict[str, float]) -> Path:
    """창 이미지 2장 + 모델 입력 + 전체 깊이맵 npz. probes.csv에 한 줄 추가."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    base = out_dir / stamp
    cv2.imwrite(f"{base}_input.png", frame.image)
    arrays: dict[str, np.ndarray] = {"input_bgr": frame.image}
    for r, c in zip(runs, canvases):
        cv2.imwrite(f"{base}_{r.spec.key}.png", c)
        arrays[f"{r.spec.key}_depth_m"] = r.depth
        arrays[f"{r.spec.key}_cells_m"] = r.cells
        arrays[f"{r.spec.key}_levels"] = r.levels
    np.savez_compressed(f"{base}_depth.npz", **arrays)

    ind, out = runs
    csv_path = out_dir / "probes.csv"
    new = not csv_path.exists()
    px, py = st.probe if st.probe is not None else ("", "")
    v_in = f"{probe_value(ind.depth, st.probe):.3f}" if st.probe is not None else ""
    v_out = f"{probe_value(out.depth, st.probe):.3f}" if st.probe is not None else ""
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if new:
            # true_m 은 비워 둔다 — 줄자로 잰 실제 거리를 나중에 채워 보정 곡선을 만든다
            wr.writerow(["stamp", "source", "frame", "probe_x", "probe_y",
                         "indoor_probe_m", "outdoor_probe_m", "indoor_nearest_m",
                         "outdoor_nearest_m", "cell_ratio_median_out_over_in", "true_m"])
        wr.writerow([stamp, source_label, frame.index, px, py, v_in, v_out,
                     f"{ind.cells.min():.3f}", f"{out.cells.min():.3f}",
                     f"{xs['ratio_med']:.3f}", ""])
    log.info("스냅샷 저장: %s_{input,indoor,outdoor}.png, _depth.npz", base)
    return base


def print_summary(runs: list[ModelRun], n: int, elapsed: float, source_label: str,
                  ratio_hist: list[float], ndiff_hist: list[int]) -> None:
    print()
    print(f"── 비교 요약: {source_label} / {n} 프레임 / {elapsed:.1f}초 ──")
    for r in runs:
        if not r.all_ms:
            print(f"  {KO[r.spec.key]} 추론  측정된 프레임 없음 (유휴 직후 프레임은 제외)")
            continue
        a = np.asarray(r.all_ms)
        print(f"  {KO[r.spec.key]} 추론  평균 {a.mean():5.1f} ms  p50 {np.percentile(a, 50):5.1f}"
              f"  p95 {np.percentile(a, 95):5.1f}  (측정 {len(a)}프레임, {r.engine_path.name})")
    if n and elapsed > 0:
        print(f"  루프 (두 모델 + 표시)  {n / elapsed:.1f} fps  — 실제 장치 속도 아님")
    if ratio_hist:
        rh = np.asarray(ratio_hist)
        print(f"  셀 거리 비율 실외/실내  중앙값 x{np.median(rh):.2f}"
              f"  (프레임별 x{rh.min():.2f} ~ x{rh.max():.2f})")
        print(f"  레벨 불일치 셀  평균 {np.mean(ndiff_hist):.1f} / {geo.CELL_COUNT}")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="설정 파일 (기본 config/default.yaml)")
    ap.add_argument("--source", default=None,
                    help="카메라 번호 | 영상 | 이미지 | 이미지 폴더 (기본: 설정의 카메라)")
    ap.add_argument("--indoor-engine", default=None, help="실내 엔진 경로 override")
    ap.add_argument("--outdoor-engine", default=None, help="실외 엔진 경로 override")
    ap.add_argument("--headless", action="store_true",
                    help="창 없이 실행하고 마지막 결과를 저장 (원격/검증용)")
    ap.add_argument("--frames", type=int, default=0,
                    help="처리할 프레임 수 (0=무한, headless 기본 30). "
                         "headless + 이미지 폴더면 이미지당 프레임 수")
    ap.add_argument("--duration", type=float, default=0.0, help="지정 초 후 자동 종료 (0=무한)")
    ap.add_argument("--scale", type=float, default=1.0, help="창 크기 배율")
    ap.add_argument("--probe", default=None, metavar="X,Y",
                    help="처음부터 거리 비교할 지점 (모델 입력 좌표, 예: 252,140)")
    ap.add_argument("--out-dir", default="logs/compare", help="스냅샷 저장 폴더")
    return ap.parse_args()


def engine_path_for(ms: ModelSpec, override: str | None, cfg: Config,
                    spec: geo.InputSpec) -> Path:
    if override:
        return cfg.path(override)
    return REPO / "models" / f"da2_metric_{ms.tag}_vits_{spec.width}x{spec.height}.engine"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load(args.config)
    cfg.validate()  # 치명적 설정 오류만 여기서 잡는다. 경고는 run.py가 출력
    spec = cfg.depth.spec

    # Qt는 디스플레이가 없으면 예외 없이 프로세스를 abort한다 → GUI 호출 전에 확인
    if not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log.warning("DISPLAY 없음 — headless로 전환 (결과는 %s 에 저장)", args.out_dir)
        args.headless = True
    if args.headless and args.frames <= 0 and args.duration <= 0:
        args.frames = 30

    runs: list[ModelRun] = []
    for ms, override, flag in ((INDOOR, args.indoor_engine, ""),
                               (OUTDOOR, args.outdoor_engine, " --outdoor")):
        p = engine_path_for(ms, override, cfg, spec)
        if not p.exists():
            log.error("%s 엔진 없음: %s\n  빌드: python tools/build_engine.py --spec %s%s",
                      KO[ms.key], p, spec.name, flag)
            return 1
        dcfg = dataclasses.replace(cfg.depth, engine=str(p), max_depth_m=ms.max_depth_m)
        runs.append(ModelRun(ms, p, DepthEngine(dcfg, cfg.grid, p),
                             DepthMapper(cfg.mapping, cfg.filter, geo.CELL_COUNT)))
    for r in runs:
        r.engine.warmup(10)
    ind, out = runs

    try:
        source, kind, source_label = open_source(args.source, cfg, spec)
    except Exception as e:  # noqa: BLE001
        log.error("소스 열기 실패: %s", e)
        return 1

    st = ViewState()
    rngs = [(cfg.mapping.near_m, cfg.mapping.far_m), (0.5, 20.0), (0.5, 80.0)]
    out_dir = cfg.path(args.out_dir)
    frame_h, frame_w = spec.height, spec.width
    if args.probe:
        try:
            px, py = (int(v) for v in args.probe.split(","))
        except ValueError:
            log.error("--probe 형식은 X,Y (예: 252,140): %s", args.probe)
            source.close()
            return 1
        if not (0 <= px < frame_w and 0 <= py < frame_h):
            log.error("--probe (%d,%d)가 입력 범위 %dx%d 밖", px, py, frame_w, frame_h)
            source.close()
            return 1
        st.probe = (px, py)

    if not args.headless:
        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            yy = y - TITLE_H
            if event == cv2.EVENT_LBUTTONDOWN and 0 <= yy < frame_h and 0 <= x < 2 * frame_w:
                st.probe = (x % frame_w, yy)  # 입력 패널·깊이 패널 어느 쪽을 눌러도 같은 점
            elif event == cv2.EVENT_RBUTTONDOWN:
                st.probe = None

        for r in runs:
            cv2.namedWindow(r.win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                            | cv2.WINDOW_GUI_NORMAL)
            cv2.setMouseCallback(r.win, on_mouse)

    frame: Frame | None = None
    canvases: list[np.ndarray] = []
    xs: dict[str, float] = {}
    n_proc = 0
    per_image = 0
    saved_at = -1  # 마지막으로 저장한 n_proc (headless 종료 시 중복 저장 방지)
    misses = 0
    placed = False
    loop_dt: deque = deque(maxlen=30)
    t_prev: float | None = None
    t_last_infer: float | None = None
    ratio_hist: list[float] = []
    ndiff_hist: list[int] = []

    def label_now() -> str:
        return source.label if kind == "images" else source_label

    def render() -> list[np.ndarray]:
        fps = 1.0 / float(np.mean(loop_dt)) if loop_dt else 0.0
        return [compose(r, ind, out, frame, cfg, st, rngs[st.range_idx], xs, label_now(), fps)
                for r in runs]

    if args.headless:
        log.info("시작 — headless, 결과는 %s", out_dir)
    else:
        log.info("시작 — 키: q 종료 / space 일시정지 / s 저장 / c 색범위 / n,p 이미지 / "
                 "왼클릭 지점 거리 비교, 오른클릭 해제")
    t_start = time.perf_counter()

    try:
        while True:
            if args.duration > 0 and time.perf_counter() - t_start > args.duration:
                log.info("지정 시간 경과 — 종료")
                break

            # ── n/p 이미지 전환: 필터 상태가 이전 이미지를 끌고 오지 않게 초기화 ──
            if st.step_image and kind == "images":
                source.step(st.step_image)
                st.step_image = 0
                st.need_frame = True
                for r in runs:
                    r.mapper.reset()

            # ── 새 프레임: 같은 이미지를 두 엔진에 넣는다 ──
            if frame is None or not st.paused or st.need_frame:
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
                now = time.perf_counter()
                dt = 1 / 15 if t_prev is None else min(1.0, max(1e-3, now - t_prev))
                if t_prev is not None:
                    loop_dt.append(now - t_prev)
                t_prev = now

                # 실행 순서를 프레임마다 바꿔 순서에 따른 시간 편향을 없앤다.
                # 저장·일시정지 직후처럼 GPU가 쉬었던 프레임은 시간 통계에서 제외.
                timed = t_last_infer is not None and now - t_last_infer < IDLE_GAP_S
                for r in (runs if n_proc % 2 == 0 else runs[::-1]):
                    t0 = time.perf_counter()
                    r.cells = r.engine.infer(frame.image)  # 내부에서 스트림 동기화
                    if timed:
                        ms = (time.perf_counter() - t0) * 1000
                        r.recent_ms.append(ms)
                        r.all_ms.append(ms)
                t_last_infer = time.perf_counter()
                for r in runs:
                    r.depth = r.engine.last_depth_map()
                    r.levels = r.mapper.update(r.cells, dt=dt)

                xs = cross_stats(ind, out)
                ratio_hist.append(xs["ratio_med"])
                ndiff_hist.append(xs["ndiff"])
                n_proc += 1
                per_image += 1
                st.need_frame = False

            # ── headless: 저장할 때만 그린다 ──
            if args.headless:
                done_image = kind == "images" and args.frames > 0 and per_image >= args.frames
                done_all = kind != "images" and args.frames > 0 and n_proc >= args.frames
                if done_image or done_all:
                    save_snapshot(out_dir, runs, render(), frame, st, label_now(), xs)
                    saved_at = n_proc
                    if done_all or source.i + 1 >= source.count:
                        break
                    st.step_image = 1  # 폴더의 다음 이미지
                    per_image = 0
                continue

            # ── 창 표시 ──
            canvases = render()
            for r, c in zip(runs, canvases):
                cv2.imshow(r.win, c)
            if not placed:
                ch_, cw_ = canvases[0].shape[:2]
                place_windows(runs, int(cw_ * args.scale), int(ch_ * args.scale))
                placed = True

            key = cv2.waitKey(30 if st.paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                st.paused = not st.paused
                t_prev = None  # 정지 시간이 fps·필터 dt에 섞이지 않게
                loop_dt.clear()
            elif key == ord("s"):
                save_snapshot(out_dir, runs, canvases, frame, st, label_now(), xs)
            elif key == ord("c"):
                st.range_idx = (st.range_idx + 1) % len(rngs)
            elif key in (ord("n"), ord("p")) and kind == "images":
                st.step_image = 1 if key == ord("n") else -1

            # 창 닫기 버튼(X)으로 닫으면 종료
            if any(cv2.getWindowProperty(r.win, cv2.WND_PROP_VISIBLE) < 1 for r in runs):
                break
            if args.frames > 0 and n_proc >= args.frames:
                break
    except KeyboardInterrupt:
        log.info("Ctrl+C — 종료")
    finally:
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()

    # --duration 으로 끝난 headless 실행도 마지막 결과를 남긴다
    if args.headless and frame is not None and saved_at != n_proc:
        save_snapshot(out_dir, runs, render(), frame, st, label_now(), xs)

    print_summary(runs, n_proc, time.perf_counter() - t_start, source_label,
                  ratio_hist, ndiff_hist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
