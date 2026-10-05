"""디버그 시각화. 촉각 출력을 눈으로 확인할 수 있게 만든다."""

from __future__ import annotations

import cv2
import numpy as np

from . import geometry as geo
from .config import Config
from .pipeline import FrameResult

# 레벨별 색 (0=먼 파랑 → 7=가까운 빨강)
_LEVEL_LUT = np.array(
    [(140, 60, 20), (150, 100, 20), (140, 150, 30), (80, 170, 60),
     (40, 180, 140), (40, 160, 220), (50, 100, 240), (60, 60, 255)],
    dtype=np.uint8,
)


def level_color(level: int, levels: int) -> tuple[int, int, int]:
    idx = int(round(level / max(1, levels - 1) * (len(_LEVEL_LUT) - 1)))
    b, g, r = _LEVEL_LUT[min(idx, len(_LEVEL_LUT) - 1)]
    return int(b), int(g), int(r)


def colorize_depth(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    """거리 고정 범위로 색을 매긴다. 프레임별 정규화를 쓰지 않는 이유와 동일."""
    d = np.clip(depth_m, near, far)
    norm = ((far - d) / (far - near) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def draw_tactile_pad(
    levels: np.ndarray, hazards: np.ndarray, cfg: Config,
    cell_px: int = 78, margin: int = 14,
) -> np.ndarray:
    """
    실제 촉각 패드의 물리 배치를 그린다. 셀 크기는 실측 피치(30×40mm) 비율.
    왼손이 좌측 3열, 오른손이 우측 3열을 담당한다.
    """
    R, C = geo.GRID_ROWS, geo.GRID_COLS
    aspect = geo.ROW_PITCH_MM / geo.COL_PITCH_MM  # 40/30 = 1.333
    cw, chh = cell_px, int(cell_px * aspect)
    W = C * cw + margin * 2
    H = R * chh + margin * 2 + 42
    img = np.full((H, W), 26, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    prof = cfg.mapping.profile
    for r in range(R):
        for c in range(C):
            i = r * C + c
            lv = int(levels[i])
            x0, y0 = margin + c * cw, margin + r * chh
            col = level_color(lv, cfg.mapping.levels)
            # 레벨에 비례한 원 크기 = 핀 돌출 높이
            frac = lv / max(1, cfg.mapping.levels - 1)
            rad = int(6 + frac * (min(cw, chh) * 0.42 - 6))
            cx, cy = x0 + cw // 2, y0 + chh // 2
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + cw - 2, y0 + chh - 2),
                          (55, 55, 55), 1)
            cv2.circle(img, (cx, cy), rad, col, -1)
            if hazards[i] > 0:
                # 위험 셀은 이중 링으로 표시 (실제로는 진동)
                cv2.circle(img, (cx, cy), rad + 5, (0, 215, 255), 2)
                cv2.circle(img, (cx, cy), rad + 9, (0, 215, 255), 1)
            cv2.putText(img, f"{lv}", (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 250, 250), 1, cv2.LINE_AA)
            cv2.putText(img, f"{prof.level_to_mm(lv):.0f}mm", (x0 + 5, y0 + chh - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)

    # 좌우 손 경계
    xm = margin + (C // 2) * cw
    cv2.line(img, (xm, margin), (xm, margin + R * chh), (0, 255, 255), 2)
    cv2.putText(img, "LEFT PALM", (margin + 6, H - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "RIGHT PALM", (xm + 6, H - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img,
                f"pitch {geo.COL_PITCH_MM:.0f}x{geo.ROW_PITCH_MM:.0f}mm  "
                f"stroke {cfg.mapping.stroke_mm:.1f}mm  step {prof.step_mm:.2f}mm",
                (margin, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (150, 150, 150), 1, cv2.LINE_AA)
    return img


def draw_depth_panel(r: FrameResult, depth_m: np.ndarray, cfg: Config) -> np.ndarray:
    """깊이맵 + 셀 경계 + 셀 거리 수치 + 검출 박스. 색 범위는 지금 쓰는 d_far까지."""
    far = r.scene.far_m if r.scene is not None else cfg.mapping.far_m
    vis = colorize_depth(depth_m, cfg.mapping.near_m, far)
    H, W = vis.shape[:2]
    R, C = geo.GRID_ROWS, geo.GRID_COLS
    ch, cw = H // R, W // C

    for i, d in enumerate(r.detections):
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        col = (0, 215, 255) if d["danger"] > 0 else (160, 160, 160)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1)
        cv2.putText(vis, f"{d['name']}:{d['danger']}", (x1 + 2, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

    for row in range(R):
        for c in range(C):
            i = row * C + c
            x0, y0 = c * cw, row * ch
            cv2.rectangle(vis, (x0, y0), (x0 + cw, y0 + ch), (60, 60, 60), 1)
            txt = f"{r.cell_dist[i]:.1f}m"
            cv2.putText(vis, txt, (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"L{int(r.levels[i])}", (x0 + 4, y0 + ch - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        level_color(int(r.levels[i]), cfg.mapping.levels), 1, cv2.LINE_AA)
    return vis


def compose(r: FrameResult, depth_m: np.ndarray, cfg: Config, stats_text: str) -> np.ndarray:
    """[원본 | 깊이+그리드] 위, [촉각 패드 | 통계] 아래."""
    cam = r.frame.image.copy()
    dep = draw_depth_panel(r, depth_m, cfg)
    top = np.hstack([cam, dep])

    pad = draw_tactile_pad(r.levels, r.hazards, cfg)
    panel_h = max(pad.shape[0], 200)
    info = np.full((panel_h, max(470, top.shape[1] - pad.shape[1]), 3), 26, dtype=np.uint8)

    m = cfg.mapping
    prof = m.profile
    n_haz = int(np.count_nonzero(r.hazards))
    far = r.scene.far_m if r.scene is not None else m.far_m
    head = [
        f"mapping {m.mode}  {m.near_m:.1f}-{far:.1f}m  {m.levels} levels",
        f"stroke {m.stroke_mm:.1f}mm  step {prof.step_mm:.2f}mm  (JND {geo.PALM_HEIGHT_JND_MM}mm)",
        f"nearest cell {float(r.cell_dist.min()):.2f}m   hazard cells {n_haz}",
    ]
    if r.scene is not None:
        s = r.scene
        p = "-" if s.p_smooth is None else f"{s.p_smooth * 100:.0f}%"
        head.append(f"scene {s.state}{' (unsure)' if s.uncertain else ''}  P(out) {p}"
                    f"  model {s.models_label}  d_far {s.far_m:.1f}m")
    lines = head + [""] + stats_text.split("\n")
    for k, line in enumerate(lines[:16]):
        col = (120, 220, 255) if k < len(head) else (210, 210, 210)
        cv2.putText(info, line, (10, 20 + k * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, col, 1, cv2.LINE_AA)

    if pad.shape[0] < panel_h:
        pad = np.vstack([pad, np.full((panel_h - pad.shape[0], pad.shape[1], 3), 26, np.uint8)])
    bottom = np.hstack([pad, info])

    w = max(top.shape[1], bottom.shape[1])
    def pad_w(img: np.ndarray) -> np.ndarray:
        if img.shape[1] == w:
            return img
        return np.hstack([img, np.full((img.shape[0], w - img.shape[1], 3), 26, np.uint8)])
    return np.vstack([pad_w(top), pad_w(bottom)])
