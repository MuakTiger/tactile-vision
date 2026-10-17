#!/usr/bin/env python3
"""
핀 영점(0단계) 조정 — 창에서 24칸을 골라 칸마다 0단계 높이를 1 tick씩 맞춘다.

0단계 = 가장 먼 거리 = 핀이 패드 표면 위로 나오지 않는 높이. 같은 지령에서도 3D 프린트·조립 공차로
핀마다 높이가 달라서, 칸마다 home_tick(0단계)을 옮긴다. max_tick(7단계)도 같은 양만큼 옮겨
핀이 올라가는 거리(약 9.0 mm)는 그대로 둔다. 1 tick ≈ 0.064 mm.
화면의 tick 숫자는 보정 전 기본값(home 372 / max 232)에서 옮긴 양 = 그 핀의 원래 영점 오차(부호 반대).

  python tools/pin_zero_gui.py             # Pico 연결 (핀이 움직임)
  python tools/pin_zero_gui.py --dry-run   # Pico 없이 화면만

조작 (창에서)
  마우스 클릭 / 방향키   칸 고르기
  w / s                  고른 핀 올림 / 내림 1 tick   (W / S = 5 tick)
  z                      고른 칸을 이번 실행을 시작할 때 값으로
  t                      고른 칸 시험: 0단계 → 7단계 → 0단계
  c 또는 스페이스        고른 칸 '확인' 표시 켜기·끄기
  0 / 7                  전체 0단계 / 전체 7단계 (칸마다 30 ms 간격)
  Enter                  저장 → calib/servo_calib.json, Pico에 바로 반영
  q / Esc                종료 (저장 안 한 변경이 있으면 한 번 더 묻는다). 핀은 천천히 0단계로

저장할 때 처음 한 번 원래 파일을 calib/servo_calib_before_zero_<월일_시분>.json으로 복사한다.
저장하지 않고 끝내면 파일과 Pico 설정은 그대로다(조정 중에는 RAW_TICK으로만 움직인다).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision.calibration import DEVICE_HOME_TICK, DEVICE_MAX_TICK, CalibTable  # noqa: E402
from tactile_vision.config import load  # noqa: E402
from tactile_vision.pipeline import Pipeline  # noqa: E402
from tactile_vision.transport import PicoLink  # noqa: E402

log = logging.getLogger("pin_zero")
WIN = "pin zero"
MAX_SHIFT = 80  # 보정 전 기본값(372/232)에서 이 이상(≈5.2 mm, 피니언 이 하나 ≈ 65 tick)은 옮기지 않는다
TICK_HARD = (82, 533)  # 펌웨어 TICK_HARD_MIN/MAX
FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_B = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

W, H = 1180, 760
CELL_W, CELL_H, GAP, HAND_GAP = 122, 120, 8, 26
GRID_X, GRID_Y = 24, 108
PANEL_X = 850
ARROWS = {65361: (0, -1), 65363: (0, 1), 65362: (-1, 0), 65364: (1, 0),  # GTK
          2424832: (0, -1), 2555904: (0, 1), 2490368: (-1, 0), 2621440: (1, 0)}  # Win/Qt

C_BG, C_INK, C_MUTED = (245, 246, 248), (30, 33, 38), (110, 116, 126)
C_CELL, C_LINE = (255, 255, 255), (200, 204, 210)
C_SEL, C_CHG, C_OK, C_UP = (37, 99, 235), (255, 243, 214), (220, 246, 228), (255, 228, 228)


class KeepAlive:
    """조정 중에 Pico failsafe(0.5 s 무신호 → 전체 홈)가 걸리지 않게 0.2 s마다 KEEPALIVE."""

    def __init__(self, link: PicoLink | None) -> None:
        self.link, self._stop, self._t = link, threading.Event(), None

    def __enter__(self) -> KeepAlive:
        if self.link is not None:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            self.link.send_keepalive()
            for line in self.link.drain():
                if line.startswith("ERR"):
                    log.warning("Pico: %s", line)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)


class ZeroTool:
    def __init__(self, cfg, table: CalibTable, path: Path, link: PicoLink | None) -> None:
        self.cfg, self.t, self.path, self.link = cfg, table, path, link
        self.levels = cfg.mapping.levels
        self.orig = [(c.home_tick, c.max_tick) for c in table.channels]
        self.sel = 9
        self.lv = [0] * geo.CELL_COUNT  # 각 칸에 지금 보낸 단계 (0 또는 7, 시험 중 제외)
        self.checked: set[int] = set()
        self.dirty = False
        self.backup: Path | None = None
        self.msg = "칸을 고르고 w / s로 0단계 높이를 맞추세요"
        self.confirm_quit = False
        self.click: tuple[int, int] | None = None
        self.f = {s: ImageFont.truetype(FONT, s) for s in (13, 15, 17)}
        self.fb = {s: ImageFont.truetype(FONT_B, s) for s in (17, 20, 24)}
        self.buttons = self._layout_buttons()

    # ── 계산 ──
    def up_sign(self, i: int) -> int:
        """핀을 올리는 tick 방향 (역방향 장착이면 -1)."""
        c = self.t.channels[i]
        return 1 if c.max_tick > c.home_tick else -1

    def shift(self, i: int) -> int:
        """보정 전 기본값(372) 대비 0단계 높이 변화(tick, + = 핀을 올림)."""
        return (self.t.channels[i].home_tick - DEVICE_HOME_TICK) * self.up_sign(i)

    def moved(self, i: int) -> int:
        """이번 실행에서 옮긴 양(tick, + = 핀을 올림)."""
        return (self.t.channels[i].home_tick - self.orig[i][0]) * self.up_sign(i)

    def mm_per_tick(self, i: int) -> float:
        c = self.t.channels[i]
        return c.span_mm / c.span_ticks if c.span_ticks else 0.0

    # ── 하드웨어 ──
    def raw(self, i: int, tick: int) -> None:
        if self.link is not None:
            self.link.send_raw_tick(i, tick)

    def adjust(self, n: int) -> None:
        i, c = self.sel, self.t.channels[self.sel]
        new = max(-MAX_SHIFT, min(MAX_SHIFT, self.shift(i) + n))
        d = (new - self.shift(i)) * self.up_sign(i)
        if not d:
            self.msg = f"칸 {i}: 보정 전 기본값에서 ±{MAX_SHIFT} tick까지만 옮깁니다"
            log.info("칸 %d 한계 ±%d tick에 걸림 (더 %s)", i, MAX_SHIFT, "올리려 함" if n > 0 else "내리려 함")
            return
        if not all(TICK_HARD[0] <= v + d <= TICK_HARD[1] for v in (c.home_tick, c.max_tick)):
            self.msg = f"칸 {i}: tick 한계 {TICK_HARD[0]}~{TICK_HARD[1]} 밖이라 옮기지 않음"
            return
        c.home_tick += d
        c.max_tick += d
        self.raw(i, c.home_tick)
        self.lv[i] = 0
        self.dirty = True
        self.msg = (f"칸 {i}: 0단계 {'올림' if n > 0 else '내림'} → 보정 전 대비 {self.shift(i):+d} tick "
                    f"({self.shift(i) * self.mm_per_tick(i):+.2f} mm)")
        log.info("칸 %d home %d max %d (보정 전 대비 %+d, 이번 %+d tick)", i, c.home_tick, c.max_tick,
                 self.shift(i), self.moved(i))

    def reset_cell(self) -> None:
        i, c = self.sel, self.t.channels[self.sel]
        c.home_tick, c.max_tick = self.orig[i]
        self.raw(i, c.home_tick)
        self.lv[i] = 0
        self.dirty = any(self.moved(k) for k in range(geo.CELL_COUNT))
        self.msg = f"칸 {i}: 이번 실행 시작 값({c.home_tick} / {c.max_tick})으로 되돌림"

    def test_cell(self) -> None:
        i, c = self.sel, self.t.channels[self.sel]
        self.msg = f"칸 {i} 시험 중: 0 → 7 → 0단계"
        self.show()
        seq = list(range(self.levels)) + list(range(self.levels - 2, -1, -1))
        for lv in seq:
            self.raw(i, c.tick_for_level(lv, self.levels))
            cv2.waitKey(120)
        self.lv[i] = 0
        self.msg = f"칸 {i} 시험 끝 (지금 0단계)"

    def all_level(self, lv: int) -> None:
        self.msg = f"전체 {lv}단계로 옮기는 중"
        self.show()
        for i, c in enumerate(self.t.channels):
            self.raw(i, c.tick_for_level(lv, self.levels))
            self.lv[i] = lv
            cv2.waitKey(30)  # 24개가 한꺼번에 움직이지 않게
        self.msg = f"전체 {lv}단계" + (" — 캘리퍼스로 잴 때 쓰세요" if lv else "")

    def save(self) -> None:
        errs = self.t.validate()
        if self.backup is None and self.path.exists():
            self.backup = self.path.with_name(f"servo_calib_before_zero_{time.strftime('%m%d_%H%M')}.json")
            if not self.backup.exists():
                shutil.copy2(self.path, self.backup)
        stamp = time.strftime("%Y-%m-%d %H:%M")
        for i, c in enumerate(self.t.channels):
            s = self.shift(i)
            c.note = (f"영점 보정 {stamp}: 보정 전({DEVICE_HOME_TICK}/{DEVICE_MAX_TICK}) 대비 0단계 {s:+d} tick"
                      f"({s * self.mm_per_tick(i):+.2f} mm, + = 핀을 올림), home·max 같은 양 이동 (tools/pin_zero_gui.py)")
        prev = self.t.notes.split("보정 전 파일: ")[-1] if "보정 전 파일: " in self.t.notes else None
        self.t.notes = (f"핀 영점 보정 {stamp} (tools/pin_zero_gui.py): 칸마다 0단계가 패드 표면과 같아지게 "
                        f"home_tick을 옮기고 max_tick도 같은 양만큼 옮김(스트로크 유지). 보정 전 기본값 "
                        f"{DEVICE_HOME_TICK}/{DEVICE_MAX_TICK}. 보정 전 파일: "
                        f"{prev or (self.backup.name if self.backup else '없음')}")
        self.t.save(self.path)
        if self.link is not None:
            self.link.push_calibration(self.t)
        self.dirty = False
        moved = [f"{i}:{self.shift(i):+d}" for i in range(geo.CELL_COUNT) if self.shift(i)]
        log.info("저장 %s — 보정 전 대비 %s", self.path, ", ".join(moved) or "없음")
        self.msg = f"저장함 ({self.path.name}, 보정한 칸 {len(moved)}개)" + ("  경고: " + errs[0] if errs else "")

    # ── 화면 ──
    def cell_rect(self, i: int) -> tuple[int, int, int, int]:
        r, col = divmod(i, geo.GRID_COLS)
        x = GRID_X + col * (CELL_W + GAP) + (HAND_GAP if col >= geo.GRID_COLS // 2 else 0)
        y = GRID_Y + r * (CELL_H + GAP)
        return x, y, x + CELL_W, y + CELL_H

    def _layout_buttons(self) -> list[tuple[tuple[int, int, int, int], str, str]]:
        rows = [[("w", "▲ 올림 1"), ("s", "▼ 내림 1")], [("W", "▲ 올림 5"), ("S", "▼ 내림 5")],
                [("z", "이 칸 되돌리기"), ("t", "이 칸 시험 0-7-0")], [("c", "확인 표시"), ("0", "전체 0단계")],
                [("7", "전체 7단계"), ("\r", "저장")]]
        out, y = [], 330
        for row in rows:
            x = PANEL_X
            for key, label in row:
                out.append(((x, y, x + 145, y + 44), key, label))
                x += 155
            y += 54
        return out

    def render(self) -> np.ndarray:
        im = Image.new("RGB", (W, H), C_BG)
        d = ImageDraw.Draw(im)
        d.text((GRID_X, 18), "핀 영점 조정 — 0단계(핀이 패드 위로 안 나오는 높이) 맞추기", font=self.fb[24], fill=C_INK)
        src = "Pico 연결" if self.link is not None else "Pico 없음 (화면만)"
        d.text((GRID_X, 54), f"{src} · 칸의 숫자 = 보정 전(372) 대비 · 1 tick ≈ {self.mm_per_tick(self.sel):.3f} mm"
               " · 7단계도 같은 양 이동",
               font=self.f[15], fill=C_MUTED)
        lx = (self.cell_rect(0)[0] + self.cell_rect(2)[2]) // 2
        rx = (self.cell_rect(3)[0] + self.cell_rect(5)[2]) // 2
        for x, t in ((lx, "왼손"), (rx, "오른손")):
            d.text((x, GRID_Y - 14), t, font=self.f[13], fill=C_MUTED, anchor="mm")
        for i, c in enumerate(self.t.channels):
            x0, y0, x1, y1 = self.cell_rect(i)
            s = self.shift(i)
            fill = C_OK if i in self.checked else C_UP if self.lv[i] else C_CHG if self.moved(i) else C_CELL
            d.rounded_rectangle((x0, y0, x1, y1), 8, fill=fill,
                                outline=C_SEL if i == self.sel else C_LINE, width=4 if i == self.sel else 1)
            d.text((x0 + 10, y0 + 8), f"칸 {i}", font=self.fb[20], fill=C_INK)
            if i in self.checked:
                d.text((x1 - 10, y0 + 12), "확인", font=self.f[13], fill=(22, 128, 61), anchor="ra")
            elif self.lv[i]:
                d.text((x1 - 10, y0 + 12), f"{self.lv[i]}단계", font=self.f[13], fill=(185, 28, 28), anchor="ra")
            d.text((x0 + 10, y0 + 42), f"{s:+d} tick" if s else "그대로", font=self.fb[17],
                   fill=C_INK if s else C_MUTED)
            d.text((x0 + 10, y0 + 66), f"{s * self.mm_per_tick(i):+.2f} mm" if s else " ", font=self.f[15], fill=C_MUTED)
            d.text((x0 + 10, y0 + 94), f"{c.home_tick} / {c.max_tick}", font=self.f[13], fill=C_MUTED)
        # 오른쪽: 고른 칸
        c = self.t.channels[self.sel]
        s = self.shift(self.sel)
        y = GRID_Y
        d.text((PANEL_X, y), f"고른 칸 {self.sel}", font=self.fb[24], fill=C_INK)
        r, col = divmod(self.sel, geo.GRID_COLS)
        lines = [f"위에서 {r + 1}째 줄, 왼쪽에서 {col + 1}째 칸 ({'왼손' if col < 3 else '오른손'})",
                 f"0단계 tick {c.home_tick}  (이번 시작 {self.orig[self.sel][0]}, 보정 전 {DEVICE_HOME_TICK})",
                 f"7단계 tick {c.max_tick}  (이번 시작 {self.orig[self.sel][1]}, 보정 전 {DEVICE_MAX_TICK})",
                 f"보정 전 대비 {s:+d} tick = {s * self.mm_per_tick(self.sel):+.2f} mm (한계 ±{MAX_SHIFT})",
                 f"스트로크 {c.span_mm:.2f} mm (그대로)"]
        for k, t in enumerate(lines):
            d.text((PANEL_X, y + 38 + k * 26), t, font=self.f[15], fill=C_INK if k == 3 else C_MUTED)
        moved = sum(1 for i in range(geo.CELL_COUNT) if self.moved(i))
        d.text((PANEL_X, y + 182), f"이번에 옮긴 칸 {moved}개 · 확인 {len(self.checked)}/24 · "
               f"{'저장 안 함' if self.dirty else '저장됨'}", font=self.fb[17],
               fill=(185, 28, 28) if self.dirty else (22, 128, 61))
        for (bx0, by0, bx1, by1), key, label in self.buttons:
            save = key == "\r"
            d.rounded_rectangle((bx0, by0, bx1, by1), 8, fill=C_SEL if save else (255, 255, 255),
                                outline=C_SEL if save else C_LINE, width=1)
            d.text(((bx0 + bx1) // 2, (by0 + by1) // 2), label, font=self.f[15],
                   fill=(255, 255, 255) if save else C_INK, anchor="mm")
        help_ = ("클릭·방향키 칸 고르기   w/s 1 tick   W/S 5 tick   z 되돌리기   t 시험   c 확인   "
                 "0/7 전체 단계   Enter 저장   q 종료")
        d.text((GRID_X, H - 70), help_, font=self.f[13], fill=C_MUTED)
        d.text((GRID_X, H - 40), self.msg, font=self.fb[17], fill=C_INK)
        if self.confirm_quit:
            d.rectangle((0, 0, W, H), fill=None)
            d.rounded_rectangle((300, 280, 880, 440), 12, fill=(255, 255, 255), outline=(185, 28, 28), width=3)
            d.text((590, 325), "저장하지 않은 변경이 있어요", font=self.fb[24], fill=(185, 28, 28), anchor="mm")
            d.text((590, 380), "Enter 저장하고 종료 · q 저장 안 하고 종료 · Esc 돌아가기", font=self.f[17],
                   fill=C_INK, anchor="mm")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def show(self) -> None:
        cv2.imshow(WIN, self.render())
        cv2.waitKey(1)

    # ── 입력 ──
    def on_mouse(self, ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def handle_click(self, x: int, y: int) -> str | None:
        for i in range(geo.CELL_COUNT):
            x0, y0, x1, y1 = self.cell_rect(i)
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.sel = i
                self.msg = f"칸 {i} 고름"
                return None
        for (bx0, by0, bx1, by1), key, _ in self.buttons:
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                return key
        return None

    def do(self, key: str) -> bool:
        """키 하나 처리. False = 끝내기."""
        if self.confirm_quit:
            if key == "\r":
                self.save()
                return False
            if key == "q":
                return False
            if key == "\x1b":
                self.confirm_quit = False
                self.msg = "계속합니다"
            return True
        if key in ("q", "\x1b"):
            if self.dirty:
                self.confirm_quit = True
                return True
            return False
        steps = {"w": 1, "s": -1, "W": 5, "S": -5}
        if key in steps:
            self.adjust(steps[key])
        elif key == "z":
            self.reset_cell()
        elif key == "t":
            self.test_cell()
        elif key in ("c", " "):
            self.checked ^= {self.sel}
            self.msg = f"칸 {self.sel} {'확인' if self.sel in self.checked else '확인 지움'}"
        elif key in ("0", "7"):
            self.all_level(int(key))
        elif key == "\r":
            self.save()
        return True

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKeyEx(30)
            key = None
            if self.click is not None:
                key = self.handle_click(*self.click)
                self.click = None
            elif k in ARROWS and not self.confirm_quit:
                dr, dc = ARROWS[k]
                r, col = divmod(self.sel, geo.GRID_COLS)
                self.sel = ((r + dr) % geo.GRID_ROWS) * geo.GRID_COLS + (col + dc) % geo.GRID_COLS
                self.msg = f"칸 {self.sel} 고름"
            elif k != -1 and (k & 0xFF) in (10, 13):
                key = "\r"
            elif k != -1 and 32 <= (k & 0xFF) < 127 or (k & 0xFF) == 27:
                key = chr(k & 0xFF)
            if key is not None and not self.do(key):
                break
            try:
                closed = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:  # Qt 창을 X로 닫으면 창 객체가 먼저 사라진다
                closed = True
            if closed:
                if self.dirty:
                    log.warning("창을 닫음 — 저장하지 않은 변경은 버립니다")
                break
        cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Pico 없이 화면만 (핀 안 움직임)")
    ap.add_argument("--snapshot", default=None, help="화면 한 장을 PNG로 저장하고 끝냄 (점검용, --dry-run과 함께)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load(a.config)
    path = cfg.path(cfg.calibration.file)
    table = CalibTable.load_or_default(path, levels=cfg.mapping.levels)
    if a.snapshot:
        tool = ZeroTool(cfg, table, path, None)
        tool.checked = {0, 1}
        tool.t.channels[8].home_tick -= 3
        tool.t.channels[8].max_tick -= 3
        tool.lv[15] = 7
        cv2.imwrite(a.snapshot, tool.render())
        return 0

    link = None
    if not a.dry_run:
        link = PicoLink(cfg.transport, required=True)
        Pipeline.sync_hardware(SimpleNamespace(cfg=cfg, link=link, calib=table))
    tool = ZeroTool(cfg, table, path, link)
    try:
        with KeepAlive(link):
            tool.run()
    finally:
        if link is not None:
            link.close(home_first=True)  # 펌웨어에 저장된(또는 그대로인) 0단계로 천천히
    return 0


if __name__ == "__main__":
    sys.exit(main())
