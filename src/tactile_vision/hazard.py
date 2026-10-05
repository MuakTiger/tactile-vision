"""
위험 객체 → jitter 채널 인코딩.

설계 의도
────────
촉각 24셀로는 객체 라벨을 전달할 수 없다. 핀 높이(진폭) 축은 이미 거리에
쓰이고 있으므로, 위험도는 직교한 축인 '시간 패턴(진동)'에 싣는다.

  진폭(핀 높이) = 거리        ← 연속적, 공간 정보
  진동(jitter)  = 위험도      ← 이산적, 주의 환기

손바닥의 Meissner 수용기는 20~50Hz에서 가장 민감하지만, SG90의 실측 속도
한계(부하시 약 360°/s)로는 그 대역을 낼 수 없다. v3 펌웨어는 진동을 핀 갱신
박자에 맞춘다: 박자(transport.pin_rate_hz, 기본 초당 5회)의 2·4번째 1/4 구간에
위험 셀을 비켜 놓아 거리 갱신 사이에 2번 = 박자의 2배 빈도(기본 초당 10회)로 떨린다.
진폭은 위험도 1/2/3 → hazard.jitter_deg(기본 1/2/3도). 1도는 SG90 데드밴드(약 10µs,
데이터시트 명목) 근처라 실기에서 실제로 움직이는지 확인해야 한다.

전류 제약
────────
jitter 중인 서보는 연속 이동 상태라 약 250mA를 계속 끈다. 24개가 동시에
진동하면 6A로 3S→5V 벅 정격을 넘긴다. 따라서 동시 jitter 셀 수를 상한으로
제한하고, 넘치면 가까운 것부터 자른다.
"""

from __future__ import annotations

import logging

import numpy as np

from . import geometry as geo
from .config import HazardCfg

logger = logging.getLogger(__name__)

# 위험도 → jitter 강도 (프로토콜 4bit, 0=없음). 진폭은 펌웨어가 정한다:
# v3는 SET_CONFIG로 받은 tick(기본 2/4/6 ≈ 1/2/3도), v2는 {15, 23, 31} tick ≈ {1.0, 1.5, 2.0} mm
MAX_DANGER = 3


class HazardEncoder:
    """
    YOLO 검출 박스 → 셀별 jitter 강도 (24,).

    detector가 없으면(엔진 미존재 또는 hazard.enabled=false) 항상 0을 반환하므로
    파이프라인은 위험도 없이도 동작한다.
    """

    def __init__(self, cfg: HazardCfg, in_w: int, in_h: int) -> None:
        self.cfg = cfg
        self.in_w, self.in_h = in_w, in_h
        self.model = None
        self.names: dict[int, str] = {}
        self._danger_by_id: dict[int, int] = {}

        if not cfg.enabled:
            logger.info("위험도 검출 비활성 (hazard.enabled=false)")
            return

        from pathlib import Path

        engine = Path(cfg.engine)
        if not engine.is_absolute():
            from .config import REPO_ROOT

            engine = REPO_ROOT / engine
        if not engine.exists():
            logger.warning(
                "YOLO 엔진 없음(%s) — 위험도 검출 없이 진행. "
                "거리 정보만 촉각으로 전달됩니다.", engine,
            )
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(engine), task="detect")
            # 클래스 이름 확보
            self.names = getattr(self.model, "names", None) or {}
            self._build_danger_map()
            logger.info(
                "위험도 검출 활성: %s (%d클래스 중 %d개에 위험도 지정)",
                engine.name, len(self.names), len(self._danger_by_id),
            )
        except Exception as e:  # noqa: BLE001 — 검출은 부가기능, 실패해도 계속
            logger.warning("YOLO 로드 실패(%s) — 위험도 검출 없이 진행", e)
            self.model = None

    def _build_danger_map(self) -> None:
        """설정의 클래스 이름 → 모델 클래스 id로 변환."""
        lower = {str(v).lower(): k for k, v in self.names.items()}
        unmatched = []
        for name, lvl in self.cfg.danger_levels.items():
            key = name.lower()
            if key in lower:
                self._danger_by_id[lower[key]] = max(0, min(MAX_DANGER, int(lvl)))
            else:
                unmatched.append(name)
        if unmatched:
            logger.info(
                "모델에 없는 클래스 %d개 무시: %s",
                len(unmatched), ", ".join(sorted(unmatched)[:8]),
            )

    # ── 박스 → 셀 매핑 ──
    def _boxes_to_cells(
        self, xyxy: np.ndarray, danger: np.ndarray, cell_dist: np.ndarray
    ) -> np.ndarray:
        """
        박스가 겹치는 셀에 위험도를 부여한다. 한 셀에 여러 박스가 겹치면 최대값.
        """
        R, C = geo.GRID_ROWS, geo.GRID_COLS
        cw, ch = self.in_w / C, self.in_h / R
        out = np.zeros(R * C, dtype=np.int16)

        for (x1, y1, x2, y2), d in zip(xyxy, danger):
            if d <= 0:
                continue
            c0 = max(0, min(C - 1, int(x1 // cw)))
            c1 = max(0, min(C - 1, int((x2 - 1e-6) // cw)))
            r0 = max(0, min(R - 1, int(y1 // ch)))
            r1 = max(0, min(R - 1, int((y2 - 1e-6) // ch)))
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    i = r * C + c
                    out[i] = max(out[i], d)

        # 원거리 위험은 알리지 않는다 (알림 피로 방지)
        out[cell_dist > self.cfg.max_alert_m] = 0

        # 전류 상한: 동시 jitter 셀 수를 제한하고, 넘치면 가까운 것만 남긴다
        active = np.nonzero(out > 0)[0]
        if active.size > self.cfg.max_jitter_cells:
            order = active[np.argsort(cell_dist[active])]
            keep = set(order[: self.cfg.max_jitter_cells].tolist())
            for i in active:
                if i not in keep:
                    out[i] = 0
        return out

    # ── 메인 ──
    def encode(self, bgr: np.ndarray, cell_dist: np.ndarray) -> tuple[np.ndarray, list]:
        """
        반환: (셀별 jitter 강도 (24,), 검출 박스 리스트[시각화용])
        """
        zero = np.zeros(geo.CELL_COUNT, dtype=np.int16)
        if self.model is None:
            return zero, []

        try:
            res = self.model(bgr, verbose=False, conf=self.cfg.conf, device=0)[0]
        except Exception as e:  # noqa: BLE001
            logger.warning("YOLO 추론 실패: %s", e)
            return zero, []

        if res.boxes is None or len(res.boxes) == 0:
            return zero, []

        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        danger = np.array([self._danger_by_id.get(int(c), 0) for c in cls], dtype=np.int16)

        hz = self._boxes_to_cells(xyxy, danger, cell_dist)
        dets = [
            {
                "xyxy": xyxy[i],
                "name": self.names.get(int(cls[i]), str(cls[i])),
                "danger": int(danger[i]),
                "conf": float(conf[i]),
            }
            for i in range(len(cls))
        ]
        return hz, dets

    def warmup(self, n: int = 5) -> float:
        """
        예열. ultralytics는 첫 predict 호출에서 predictor를 구성하고 TRT 엔진을
        역직렬화하므로 수 초가 걸린다. 이를 실시간 경로에 두면 초기 프레임이
        느려지고 통계가 왜곡된다(실측: 60프레임 평균이 37ms → 180ms로 튐).
        """
        import time

        if self.model is None:
            return 0.0
        dummy = np.zeros((self.in_h, self.in_w, 3), dtype=np.uint8)
        far = np.full(geo.CELL_COUNT, 1.0, dtype=np.float32)
        self.encode(dummy, far)  # 엔진 로드 + predictor 구성
        t0 = time.perf_counter()
        for _ in range(n):
            self.encode(dummy, far)
        ms = (time.perf_counter() - t0) / max(1, n) * 1000
        logger.info("위험도 검출 예열 완료: 평균 %.1fms", ms)
        return ms
