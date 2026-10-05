#!/usr/bin/env python3
"""
24채널 핀 높이 오프셋 캘리브레이션.

무엇을 하는가
────────────
3D 프린트 공차 때문에 같은 PWM tick을 넣어도 24개 핀의 높이가 다르다.
채널마다 다음 두 값을 실측해 저장한다.

  home_tick : 핀이 패드 표면과 정확히 같은 높이 (레벨 0의 기준)
  max_tick  : 기구 간섭 없이 올릴 수 있는 최대 높이 (레벨 7)

이 두 값이 정해지면 중간 레벨은 선형 보간으로 자동 계산된다.

작업 순서 (권장)
──────────────
  1단계  모든 채널의 home을 맞춘다 (h 명령으로 순회)
  2단계  모든 채널의 max를 맞춘다  (m 명령으로 순회)
  3단계  t 명령으로 전체 스윕해 균일성을 눈으로 확인
  4단계  s 명령으로 저장

안전
────
  - tick은 항상 400~2600µs(82~533tick)로 클램프된다. 그 밖은 기어가 깨진다.
  - 처음에는 작은 걸음(--step 5)으로 올리고, 저항이 느껴지면 즉시 멈출 것.
  - 스페이스바가 즉시 전체 홈 복귀다.

사용:
  python tools/calibrate_pins.py                 대화형
  python tools/calibrate_pins.py --dry-run       Pico 없이 UI 확인
  python tools/calibrate_pins.py --sweep-only    저장된 값으로 스윕만
"""

from __future__ import annotations

import argparse
import logging
import sys
import termios
import threading
import time
import tty
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision.calibration import CalibTable, ChannelCalib  # noqa: E402
from tactile_vision.config import load  # noqa: E402
from tactile_vision.transport import PicoLink  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("calib")

HELP = """
┌─ 조작 ──────────────────────────────────────────────────────────┐
│  k / j     현재 채널 tick 올림 / 내림 (step 단위)               │
│  K / J     큰 걸음 (step x 5)                                    │
│  h         현재 tick을 이 채널의 home(레벨0)으로 지정            │
│  m         현재 tick을 이 채널의 max(최대높이)으로 지정          │
│  n / p     다음 / 이전 채널                                      │
│  0-9,a-n   채널 직접 선택 (0~23)                                 │
│  g         현재 채널을 home 위치로 이동                          │
│  G         현재 채널을 max 위치로 이동                           │
│  l         현재 채널 레벨 스윕 (0→7→0)                           │
│  t         전체 채널 동시 레벨 스윕 (균일성 확인)                │
│  r         전체 raise: 모든 채널을 max로 (간섭 확인)             │
│  space     전체 홈 복귀 (천천히, 끝→끝 약 1.4 s)                 │
│  +/-       step 크기 변경                                        │
│  w         현재 채널 메모 입력                                   │
│  s         저장          q 종료(저장 확인)          ?  도움말    │
└──────────────────────────────────────────────────────────────────┘
"""

CH_KEYS = "0123456789abcdefghijklmn"  # 0~23


class Getch:
    """
    단일 문자 입력. 터미널이면 cbreak 모드, 파이프/리다이렉트면 그대로 읽는다.
    비-TTY 폴백이 있어야 스크립트로 자동 검증할 수 있다.
    """

    def __init__(self) -> None:
        self.is_tty = sys.stdin.isatty()
        self.fd = None
        self.old = None

    def __enter__(self):
        if self.is_tty:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        else:
            log.info("비대화형 입력 감지 — 표준입력에서 명령을 순차 처리합니다")
        return self

    def __exit__(self, *a):
        if self.is_tty and self.fd is not None and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self) -> str:
        c = sys.stdin.read(1)
        if not c:
            raise EOFError("입력 종료")
        return c

    def read_line(self, prompt: str) -> str:
        """메모 입력처럼 한 줄을 받는 경우."""
        if self.is_tty and self.fd is not None and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            out = input(prompt)
            tty.setcbreak(self.fd)
            return out
        print(prompt, end="", flush=True)
        return sys.stdin.readline().strip()


class KeepAlive:
    """
    Pico failsafe(500 ms 동안 프레임이 없으면 전체 홈)가 캘리브레이션 중에 핀을 내리지 않게
    0.2 s마다 KEEPALIVE를 보내고, Pico 상태 줄을 읽어 송신 버퍼가 차지 않게 한다.
    이 도구가 죽으면 keepalive도 멈추므로 failsafe는 그대로 살아 있다.
    (v2 펌웨어는 KEEPALIVE를 모르는 타입으로 보고 ERR를 내지만 failsafe 타이머는 갱신한다)
    """

    def __init__(self, link: PicoLink | None, period_s: float = 0.2) -> None:
        self.link = link
        self.period_s = period_s
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> KeepAlive:
        if self.link and self.link.connected:
            self._t = threading.Thread(target=self._run, name="keepalive", daemon=True)
            self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            self.link.send_keepalive()
            self.link.drain()

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)


class Calibrator:
    def __init__(self, link: PicoLink | None, table: CalibTable, levels: int,
                 step: int, dry: bool) -> None:
        self.link = link
        self.t = table
        self.levels = levels
        self.step = step
        self.dry = dry
        self.ch = 0
        self.tick = table.channels[0].home_tick
        self.dirty = False

    # ── 하드웨어 ──
    def apply(self, ch: int, tick: int) -> None:
        tick = max(82, min(533, tick))
        if self.link and self.link.connected:
            self.link.send_raw_tick(ch, tick)
        elif not self.dry:
            log.warning("Pico 미연결 — 명령이 전달되지 않음")

    def home_all(self) -> None:
        if self.link and self.link.connected:
            self.link.send_home()
        log.info("전체 홈 복귀")

    def cal(self, ch: int | None = None) -> ChannelCalib:
        return self.t.channels[self.ch if ch is None else ch]

    # ── 표시 ──
    def mm_of(self, ch: int, tick: int) -> float:
        """home 기준 핀 높이 추정 (mm)."""
        c = self.cal(ch)
        deg_per_tick = geo.TICK_US / ((geo.SG90_MAX_US - geo.SG90_MIN_US) / geo.SG90_RANGE_DEG)
        return (tick - c.home_tick) * deg_per_tick * geo.MM_PER_DEG

    def status(self) -> None:
        c = self.cal()
        us = self.tick * geo.TICK_US
        print("\033[2J\033[H", end="")  # 화면 지우기
        print("═" * 68)
        print(f" 핀 캘리브레이션   채널 {self.ch:2d}/23   step={self.step}tick"
              f"   {'[DRY-RUN]' if self.dry else ''}"
              f"{'' if (self.link and self.link.connected) else ' [Pico 미연결]'}")
        print("═" * 68)
        print(f"  현재 tick  {self.tick:4d}  ({us:6.0f}µs)   "
              f"추정 높이 {self.mm_of(self.ch, self.tick):+6.2f}mm")
        print(f"  저장값     home={c.home_tick:4d}  max={c.max_tick:4d}  "
              f"스팬 {c.span_mm:5.2f}mm ({c.span_deg:5.1f}°)"
              f"{'  [역방향]' if c.inverted else ''}")
        if c.note:
            print(f"  메모       {c.note}")
        print()
        # 24채널 개요
        print("  전체 스팬(mm) — 편차가 크면 기구 재정렬 필요")
        for r in range(geo.GRID_ROWS):
            row = []
            for col in range(geo.GRID_COLS):
                i = r * geo.GRID_COLS + col
                cc = self.t.channels[i]
                mark = "*" if i == self.ch else " "
                tag = "?" if "미측정" in cc.note else " "
                row.append(f"{mark}{i:2d}:{cc.span_mm:5.2f}{tag}")
            print("    " + " ".join(row))
        spans = [cc.span_mm for cc in self.t.channels]
        print(f"\n  평균 {sum(spans)/len(spans):.2f}mm  "
              f"범위 {min(spans):.2f}~{max(spans):.2f}mm  "
              f"편차 {max(spans)-min(spans):.2f}mm "
              f"(지각 한계 {geo.PALM_HEIGHT_JND_MM}mm)")
        prof = geo.StrokeProfile(sum(spans)/len(spans), self.levels)
        print(f"  {self.levels}단계 시 간격 {prof.step_mm:.2f}mm "
              f"{'OK' if prof.perceptually_valid else '← 지각 한계 미달, 단계 수를 줄일 것'}")
        print(f"\n  {'저장 필요' if self.dirty else '저장됨'}    ? 도움말   q 종료")

    # ── 동작 ──
    def sweep_channel(self) -> None:
        c = self.cal()
        print(f"\n  ch{self.ch} 레벨 스윕...")
        for lv in list(range(self.levels)) + list(range(self.levels - 2, -1, -1)):
            tk = c.tick_for_level(lv, self.levels)
            self.apply(self.ch, tk)
            print(f"    레벨{lv} → {tk}tick ({self.mm_of(self.ch, tk):5.2f}mm)")
            time.sleep(0.35)
        self.apply(self.ch, c.home_tick)

    def sweep_all(self) -> None:
        print("\n  전체 채널 동시 스윕 — 24개 핀 높이가 균일한지 확인하세요")
        for lv in list(range(self.levels)) + list(range(self.levels - 2, -1, -1)):
            for i, c in enumerate(self.t.channels):
                self.apply(i, c.tick_for_level(lv, self.levels))
                time.sleep(0.004)  # 동시 돌입전류 회피
            print(f"    레벨 {lv}")
            time.sleep(0.6)
        self.home_all()

    def raise_all(self) -> None:
        print("\n  전체 max 위치로 — 기구 간섭이 없는지 확인하세요")
        for i, c in enumerate(self.t.channels):
            self.apply(i, c.max_tick)
            time.sleep(0.03)
        time.sleep(1.5)
        self.home_all()

    def set_home(self) -> None:
        c = self.cal()
        old = c.home_tick
        c.home_tick = self.tick
        c.note = c.note.replace("미측정", "").strip() or "실측"
        self.dirty = True
        log.info("ch%d home: %d → %d", self.ch, old, self.tick)

    def set_max(self) -> None:
        c = self.cal()
        old = c.max_tick
        c.max_tick = self.tick
        c.note = c.note.replace("미측정", "").strip() or "실측"
        self.dirty = True
        log.info("ch%d max: %d → %d (스팬 %.2fmm)", self.ch, old, self.tick, c.span_mm)

    def save(self, path: Path) -> None:
        errs = self.t.validate()
        if errs:
            print("\n  검증 경고:")
            for e in errs:
                print(f"    - {e}")
        self.t.levels = self.levels
        self.t.notes = f"실측 캘리브레이션 ({time.strftime('%Y-%m-%d %H:%M')})"
        self.t.save(path)
        self.dirty = False
        print(f"\n  저장 완료: {path}")
        if self.link and self.link.connected:
            self.link.push_calibration(self.t)

    # ── 메인 루프 ──
    def run(self, path: Path) -> None:
        self.status()
        with Getch() as g:
            while True:
                try:
                    k = g.get()
                except EOFError:
                    # 비대화형 입력이 끝났다. 안전하게 홈 복귀 후 종료.
                    self.home_all()
                    print("\n  입력 종료 — 홈 복귀 후 종료"
                          + ("  (미저장 변경 있음)" if self.dirty else ""))
                    return
                if k == "q":
                    if self.dirty:
                        print("\n  저장하지 않은 변경이 있습니다. 저장할까요? (y/n) ", end="", flush=True)
                        try:
                            ans = g.get()
                        except EOFError:
                            ans = "y"  # 비대화형에서는 변경을 잃지 않도록 저장
                        if ans.lower() == "y":
                            self.save(path)
                    self.home_all()
                    print("\n  종료")
                    return
                elif k == "?":
                    print(HELP)
                    continue
                elif k == "k":
                    self.tick += self.step
                elif k == "j":
                    self.tick -= self.step
                elif k == "K":
                    self.tick += self.step * 5
                elif k == "J":
                    self.tick -= self.step * 5
                elif k == "h":
                    self.set_home()
                elif k == "m":
                    self.set_max()
                elif k == "n":
                    self.ch = (self.ch + 1) % geo.CELL_COUNT
                    self.tick = self.cal().home_tick
                elif k == "p":
                    self.ch = (self.ch - 1) % geo.CELL_COUNT
                    self.tick = self.cal().home_tick
                elif k == "g":
                    self.tick = self.cal().home_tick
                elif k == "G":
                    self.tick = self.cal().max_tick
                elif k == "l":
                    self.sweep_channel()
                elif k == "t":
                    self.sweep_all()
                elif k == "r":
                    self.raise_all()
                elif k == " ":
                    self.home_all()
                elif k == "+":
                    self.step = min(50, self.step + 1)
                elif k == "-":
                    self.step = max(1, self.step - 1)
                elif k == "s":
                    self.save(path)
                elif k == "w":
                    self.cal().note = g.read_line("\n  메모: ")
                    self.dirty = True
                elif k in CH_KEYS:
                    idx = CH_KEYS.index(k)
                    if idx < geo.CELL_COUNT:
                        self.ch = idx
                        self.tick = self.cal().home_tick
                else:
                    continue

                self.tick = max(82, min(533, self.tick))
                if k in "kjKJgG":
                    self.apply(self.ch, self.tick)
                self.status()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--step", type=int, default=5, help="tick 조정 걸음 (기본 5 ≈ 0.33mm)")
    ap.add_argument("--dry-run", action="store_true", help="Pico 없이 UI만 확인")
    ap.add_argument("--sweep-only", action="store_true", help="저장값으로 전체 스윕만 실행")
    args = ap.parse_args()

    cfg = load(args.config)
    path = cfg.path(cfg.calibration.file)
    table = CalibTable.load_or_default(path, levels=cfg.mapping.levels)

    link = None if args.dry_run else PicoLink(cfg.transport)
    if link and not link.connected:
        print("\n  Pico가 연결되지 않았습니다. UI는 동작하지만 서보는 움직이지 않습니다.")
        print("  --dry-run 으로 실행하면 이 경고가 나오지 않습니다.\n")
        time.sleep(1.5)

    c = Calibrator(link, table, cfg.mapping.levels, args.step, args.dry_run)

    try:
        with KeepAlive(link):
            if args.sweep_only:
                c.sweep_all()
                return 0
            print(HELP)
            time.sleep(0.8)
            c.run(path)
    finally:
        if link:
            link.close(home_first=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
