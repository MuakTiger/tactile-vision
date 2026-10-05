"""
채널별 서보 캘리브레이션.

왜 필요한가
──────────
랙·피니언·브래킷이 모두 3D 프린트라 24개 유닛의 공차가 다르다. 같은 PWM tick을
넣어도 핀 높이가 채널마다 다르고, 조립 시 피니언 맞물림 각도도 균일하지 않다.
단일 SAFE_MIN/SAFE_MAX로는 절대 정렬이 맞지 않는다.

기존 펌웨어는 전 채널 공통 상수(SAFE_MIN=250, SAFE_MAX=490)를 썼다. 실측하면
그 범위는 68.3°~179.3° = 111° = 15.5mm로, 가용 스트로크 20.9mm 중 5.4mm를
버리면서 홈 위치가 스트로크 중간에 놓인다.

여기서는 채널마다 home_tick(핀 최하단)과 max_tick(핀 최상단)을 따로 저장한다.
home_tick > max_tick이면 피니언이 반대로 맞물린 채널로 자동 처리된다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import geometry as geo

logger = logging.getLogger(__name__)

CALIB_VERSION = 1
TICK_MIN, TICK_MAX = 0, 4095

# 이 장치의 기본 범위 (2026-10-05). 펌웨어 tactile_core.h HOME/MAX_TICK_DEFAULT와 같아야 한다(test_firmware가 확인).
# 서보를 역방향으로 조립해 tick이 커질수록 핀이 내려간다. 사용자 결정 범위 232~372 tick(60°부터 140 tick, 9.0 mm).
DEVICE_HOME_TICK = 372  # 핀 맨 아래 (레벨 0, 먼 곳) ≈ 124.7°
DEVICE_MAX_TICK = 232  # 핀 맨 위 (가까운 곳) ≈ 60°


@dataclass
class ChannelCalib:
    ch: int
    home_tick: int  # 핀 최하단 (레벨 0)
    max_tick: int  # 핀 최상단 (레벨 levels-1)
    note: str = ""

    @property
    def inverted(self) -> bool:
        return self.max_tick < self.home_tick

    @property
    def span_ticks(self) -> int:
        return abs(self.max_tick - self.home_tick)

    @property
    def span_deg(self) -> float:
        return self.span_ticks * geo.TICK_US / (
            (geo.SG90_MAX_US - geo.SG90_MIN_US) / geo.SG90_RANGE_DEG
        )

    @property
    def span_mm(self) -> float:
        return self.span_deg * geo.MM_PER_DEG

    def tick_for_level(self, level: int, levels: int) -> int:
        """
        레벨 → tick. 펌웨어 tactile_core.h levelTick()과 같은 정수 연산이다
        (0.5는 home에서 먼 쪽으로 반올림). 두 값이 달라지면 화면과 실제 핀 위치가 어긋난다.
        """
        if levels < 2:
            raise ValueError("levels는 2 이상")
        top = levels - 1
        lv = min(max(int(level), 0), top)
        num = (self.max_tick - self.home_tick) * lv
        mag = (2 * abs(num) + top) // (2 * top)
        return self.home_tick + (mag if num >= 0 else -mag)

    def validate(self) -> list[str]:
        errs = []
        if not 0 <= self.ch < geo.CELL_COUNT:
            errs.append(f"ch{self.ch}: 채널 번호 범위 초과")
        for name, t in (("home_tick", self.home_tick), ("max_tick", self.max_tick)):
            if not TICK_MIN <= t <= TICK_MAX:
                errs.append(f"ch{self.ch}: {name}={t} 가 0~4095 밖")
            us = t * geo.TICK_US
            if not (400 <= us <= 2600):
                errs.append(
                    f"ch{self.ch}: {name}={t} → {us:.0f}µs 는 SG90 안전범위"
                    f"(400~2600µs) 밖. 기구 손상 위험"
                )
        if self.span_ticks < 20:
            errs.append(
                f"ch{self.ch}: 스팬 {self.span_ticks}tick 은 너무 좁음 "
                "(캘리브레이션 미완료로 보임)"
            )
        if self.span_mm > geo.RACK_LENGTH_MM:
            errs.append(f"ch{self.ch}: 스팬 {self.span_mm:.1f}mm 가 랙 길이 초과")
        return errs


@dataclass
class CalibTable:
    version: int = CALIB_VERSION
    created: str = ""
    levels: int = 8
    notes: str = ""
    channels: list[ChannelCalib] = field(default_factory=list)

    # ── 생성 ──
    @classmethod
    def default(cls, levels: int = 8, stroke_mm: float | None = None) -> CalibTable:
        """
        캘리브레이션 파일이 없을 때 쓰는 초기 테이블 — 이 장치의 기본 범위(펌웨어 기동 기본값과 같음).
        stroke_mm을 주면 같은 home·방향으로 그 스트로크만큼만 올린다.
        """
        home = DEVICE_HOME_TICK
        top = DEVICE_MAX_TICK
        if stroke_mm is not None:
            span_deg = geo.mm_to_deg(stroke_mm)
            if span_deg > geo.SG90_RANGE_DEG:
                raise ValueError(f"스트로크 {stroke_mm}mm는 SG90 각도 범위 초과")
            up = 1 if top > home else -1
            top = home + up * int(round(span_deg * geo.TICKS_PER_DEG))
        return cls(
            created=datetime.now().isoformat(timespec="seconds"),
            levels=levels,
            notes=(
                "장치 기본 범위 (미측정, 펌웨어 기동 기본값과 같음). "
                "tools/calibrate_pins.py 로 채널별 실측 후 교체할 것."
            ),
            channels=[
                ChannelCalib(ch=i, home_tick=home, max_tick=top, note="미측정")
                for i in range(geo.CELL_COUNT)
            ],
        )

    # ── 입출력 ──
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["_derived"] = {
            "mm_per_deg": round(geo.MM_PER_DEG, 6),
            "tick_us": round(geo.TICK_US, 4),
            "span_mm_per_channel": [round(c.span_mm, 2) for c in self.channels],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("캘리브레이션 저장: %s (%d채널)", path, len(self.channels))

    @classmethod
    def load(cls, path: Path) -> CalibTable:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != CALIB_VERSION:
            raise ValueError(
                f"캘리브레이션 버전 불일치: {raw.get('version')} != {CALIB_VERSION}"
            )
        chans = [
            ChannelCalib(
                ch=c["ch"],
                home_tick=c["home_tick"],
                max_tick=c["max_tick"],
                note=c.get("note", ""),
            )
            for c in raw["channels"]
        ]
        return cls(
            version=raw["version"],
            created=raw.get("created", ""),
            levels=raw.get("levels", 8),
            notes=raw.get("notes", ""),
            channels=chans,
        )

    @classmethod
    def load_or_default(cls, path: Path, levels: int = 8) -> CalibTable:
        if path.exists():
            t = cls.load(path)
            logger.info("캘리브레이션 로드: %s (생성 %s)", path.name, t.created)
            return t
        logger.warning(
            "캘리브레이션 파일 없음(%s) — 장치 기본 범위(home %d / max %d tick) 사용. "
            "실제 핀 높이는 채널마다 다를 것이므로 tools/calibrate_pins.py 실행 권장",
            path, DEVICE_HOME_TICK, DEVICE_MAX_TICK,
        )
        return cls.default(levels=levels)

    @classmethod
    def for_tool(cls, path: Path, levels: int, home: int | None = None, span: int | None = None,
                 note: str = "tool") -> CalibTable:
        """
        시험 도구(servo_*)용 범위 — Pico 메모리에만 넣고 파일은 바꾸지 않는다.
        기본은 캘리브레이션 파일 그대로(채널마다 home = 핀 맨 아래, max = 핀 맨 위, 방향 포함).
        home·span을 주면 모든 칸에 그 값: home = 핀 맨 아래 tick, span = 위로 올라가는 tick 수(양수).
        올라가는 방향(tick +/−)은 채널마다 파일을 따른다 — 역방향 조립(home > max)도 그대로.
        """
        base = cls.load_or_default(path, levels=levels)
        chans = []
        for c in sorted(base.channels, key=lambda c: c.ch):
            up = -1 if c.inverted else 1
            h = c.home_tick if home is None else int(home)
            s = c.span_ticks if span is None else int(span)
            chans.append(ChannelCalib(c.ch, h, h + up * s, note))
        return cls(levels=levels, notes=f"{note} 임시 범위 (파일 아님)", channels=chans)

    def ends(self) -> tuple[np.ndarray, np.ndarray]:
        """채널 순서대로 (핀 맨 아래 tick, 핀 맨 위 tick) 배열."""
        by_ch = sorted(self.channels, key=lambda c: c.ch)
        return (np.array([c.home_tick for c in by_ch], dtype=int),
                np.array([c.max_tick for c in by_ch], dtype=int))

    def range_text(self) -> str:
        """시험 도구 화면용 한 줄 설명."""
        down, up = self.ends()

        def deg(t: int) -> float:
            return (geo.tick_to_us(t) - geo.SG90_MIN_US) / (geo.SG90_MAX_US - geo.SG90_MIN_US) * geo.SG90_RANGE_DEG

        if np.all(down == down[0]) and np.all(up == up[0]):
            d, u = int(down[0]), int(up[0])
            s = abs(u - d)
            return (f"핀 맨 아래 {d} tick({deg(d):.1f}°) ↔ 맨 위 {u} tick({deg(u):.1f}°), 범위 {s} tick = "
                    f"{s / geo.TICKS_PER_DEG:.1f}° = {s / geo.TICKS_PER_DEG * geo.MM_PER_DEG:.1f} mm"
                    + (" (역방향 조립: tick이 작을수록 위)" if u < d else ""))
        return (f"채널별 캘리브레이션 범위 (맨 아래 {down.min()}~{down.max()}, 맨 위 {up.min()}~{up.max()} tick)")

    # ── 검증 ──
    def validate(self) -> list[str]:
        errs: list[str] = []
        if len(self.channels) != geo.CELL_COUNT:
            errs.append(f"채널 수 {len(self.channels)} != {geo.CELL_COUNT}")
        seen = set()
        for c in self.channels:
            if c.ch in seen:
                errs.append(f"채널 {c.ch} 중복")
            seen.add(c.ch)
            errs.extend(c.validate())

        # 채널간 스트로크 편차 — 너무 크면 촉각 패턴이 왜곡된다
        if self.channels:
            spans = np.array([c.span_mm for c in self.channels])
            spread = float(spans.max() - spans.min())
            if spread > 3.0:
                errs.append(
                    f"채널간 스트로크 편차 {spread:.1f}mm — 손바닥 식별 한계"
                    f"({geo.PALM_HEIGHT_JND_MM}mm) 초과. 기구 재정렬 필요"
                )
        return errs

    # ── 사용 ──
    def tick_matrix(self, levels: int | None = None) -> np.ndarray:
        """(24, levels) tick 룩업 테이블. 펌웨어 전송/시뮬레이션용."""
        n = levels if levels is not None else self.levels
        out = np.zeros((geo.CELL_COUNT, n), dtype=np.int16)
        by_ch = {c.ch: c for c in self.channels}
        for ch in range(geo.CELL_COUNT):
            c = by_ch[ch]
            for lv in range(n):
                out[ch, lv] = c.tick_for_level(lv, n)
        return out

    def summary(self) -> str:
        if not self.channels:
            return "채널 없음"
        spans = np.array([c.span_mm for c in self.channels])
        inv = [c.ch for c in self.channels if c.inverted]
        unmeasured = [c.ch for c in self.channels if "미측정" in c.note]
        lines = [
            f"캘리브레이션 v{self.version} ({self.created}) {self.levels}단계",
            f"  스트로크 평균 {spans.mean():.2f}mm  범위 {spans.min():.2f}~{spans.max():.2f}mm"
            f"  편차 {spans.max()-spans.min():.2f}mm",
            f"  단계 간격 평균 {spans.mean()/(self.levels-1):.2f}mm "
            f"(지각 한계 {geo.PALM_HEIGHT_JND_MM}mm)",
        ]
        if inv:
            lines.append(f"  역방향 채널: {inv}")
        if unmeasured:
            lines.append(f"  미측정 채널 {len(unmeasured)}개: {unmeasured}")
        return "\n".join(lines)
