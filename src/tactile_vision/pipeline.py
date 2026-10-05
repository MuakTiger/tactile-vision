"""
메인 파이프라인.

  카메라 → 크롭/리사이즈 → [장면 판별 → 모델·d_far 선택] → metric depth(TRT) → 셀 풀링(GPU)
        → 비대칭 시간필터 → 레벨 양자화 → 위험도 jitter → 바이너리 전송 → Pico

장면 전환(scene.mode = auto)이면 실내·실외 깊이 엔진과 Places365 판별기를 함께 올리고,
프레임마다 SceneSwitcher가 고른 모델만 돌린다. 애매하면 두 모델의 셀별 최소(가까운 값).

단계별 소요시간을 상시 계측한다. 졸업논문의 지연 예산 표를 이 계측값으로 채운다.
"""

from __future__ import annotations

import csv
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry as geo
from .calibration import CalibTable
from .camera import Camera, Frame, VideoFileSource
from .config import Config
from .depth import DepthEngine
from .hazard import HazardEncoder
from .mapping import DepthMapper
from .scene import SceneClassifier
from .switching import KO as SCENE_KO
from .switching import SceneDecision, SceneSwitcher
from .transport import PicoLink

logger = logging.getLogger(__name__)

# scene.mode → 올릴 깊이 엔진
ENGINES_FOR_MODE = {"off": ("indoor",), "indoor": ("indoor",),
                    "outdoor": ("outdoor",), "auto": ("indoor", "outdoor")}


@dataclass
class StageTimes:
    """한 프레임의 단계별 소요시간 (ms)."""

    capture_age: float = 0.0  # 캡처 시각 ~ 처리 시작 (프레임 신선도)
    scene: float = 0.0  # 장면 판별 + 전환 결정 (판별하지 않은 프레임은 ~0)
    depth: float = 0.0  # 깊이 추론 + 셀 풀링. 두 모델을 돌린 프레임은 합
    hazard: float = 0.0
    filt: float = 0.0
    tx: float = 0.0
    total: float = 0.0

    def as_row(self) -> list[float]:
        return [
            round(self.capture_age, 2), round(self.scene, 2), round(self.depth, 2),
            round(self.hazard, 2), round(self.filt, 2), round(self.tx, 2), round(self.total, 2),
        ]


@dataclass
class FrameResult:
    frame: Frame
    cell_dist: np.ndarray  # 셀 원거리 (미터)
    levels: np.ndarray  # 양자화 레벨
    hazards: np.ndarray  # jitter 강도
    detections: list = field(default_factory=list)
    times: StageTimes = field(default_factory=StageTimes)
    sent: bool = False
    scene: SceneDecision | None = None  # scene.mode = off면 None


class Stats:
    """이동 창 통계. 벤치마크와 실시간 표시에 함께 쓴다."""

    def __init__(self, window: int = 120) -> None:
        self.window = window
        self.fps = deque(maxlen=window)
        self.depth_ms = deque(maxlen=window)
        self.total_ms = deque(maxlen=window)
        self.age_ms = deque(maxlen=window)
        self.n_frames = 0
        self.n_sent = 0
        self.level_changes = 0
        self._prev_levels: np.ndarray | None = None
        self.t_start = time.perf_counter()
        # 장면 (scene.mode != off)
        self.scene_frames = {"indoor": 0, "outdoor": 0}
        self.n_both = 0
        self.n_switches = 0
        self.scene_ms = deque(maxlen=window)  # 판별한 프레임만
        self._prev_state: str | None = None

    def update(self, r: FrameResult) -> None:
        self.n_frames += 1
        if r.sent:
            self.n_sent += 1
        t = r.times
        if t.total > 0:
            self.fps.append(1000.0 / t.total)
        self.depth_ms.append(t.depth)
        self.total_ms.append(t.total)
        self.age_ms.append(t.capture_age)
        if self._prev_levels is not None:
            self.level_changes += int(np.count_nonzero(r.levels != self._prev_levels))
        self._prev_levels = r.levels.copy()
        d = r.scene
        if d is not None:
            self.scene_frames[d.state] += 1
            self.n_both += len(d.models) > 1
            if self._prev_state is not None and d.state != self._prev_state:
                self.n_switches += 1
            self._prev_state = d.state
            if d.classified:
                self.scene_ms.append(d.clf_ms)

    def _scene_share(self) -> tuple[float, float, float] | None:
        n = sum(self.scene_frames.values())
        if n == 0:
            return None
        return (100 * self.scene_frames["indoor"] / n, 100 * self.scene_frames["outdoor"] / n,
                100 * self.n_both / n)

    @staticmethod
    def _p(d: deque, q: float) -> float:
        return float(np.percentile(d, q)) if d else 0.0

    def summary(self) -> str:
        el = time.perf_counter() - self.t_start
        moves_per_min = self.level_changes / max(el, 1e-6) * 60 / geo.CELL_COUNT
        return (
            f"프레임 {self.n_frames} / 전송 {self.n_sent} / {el:.1f}초\n"
            f"  FPS        평균 {np.mean(self.fps) if self.fps else 0:.1f}  "
            f"p50 {self._p(self.fps,50):.1f}  p5 {self._p(self.fps,5):.1f}\n"
            f"  깊이추론   평균 {np.mean(self.depth_ms) if self.depth_ms else 0:.1f}ms  "
            f"p95 {self._p(self.depth_ms,95):.1f}ms\n"
            f"  전체지연   평균 {np.mean(self.total_ms) if self.total_ms else 0:.1f}ms  "
            f"p95 {self._p(self.total_ms,95):.1f}ms\n"
            f"  프레임나이 평균 {np.mean(self.age_ms) if self.age_ms else 0:.1f}ms  "
            f"p95 {self._p(self.age_ms,95):.1f}ms\n"
            f"  서보이동   {moves_per_min:.1f}회/분/서보"
            + self._scene_line_ko()
        )

    def _scene_line_ko(self) -> str:
        s = self._scene_share()
        if s is None:
            return ""
        clf = f"{np.mean(self.scene_ms):.1f}ms" if self.scene_ms else "-"
        return (f"\n  장면       실내 {s[0]:.0f}% / 실외 {s[1]:.0f}% / 두 모델 {s[2]:.0f}%  "
                f"전환 {self.n_switches}회  판별 평균 {clf}")

    def summary_ascii(self) -> str:
        """
        OpenCV 오버레이용. cv2.putText는 한글 글리프를 렌더링하지 못해
        '?????'로 표시되므로 디버그 화면에는 ASCII만 쓴다.
        """
        el = time.perf_counter() - self.t_start
        moves = self.level_changes / max(el, 1e-6) * 60 / geo.CELL_COUNT
        return (
            f"frames {self.n_frames}  sent {self.n_sent}  {el:.1f}s\n"
            f"FPS    avg {np.mean(self.fps) if self.fps else 0:5.1f}"
            f"  p50 {self._p(self.fps,50):5.1f}  p5 {self._p(self.fps,5):5.1f}\n"
            f"depth  avg {np.mean(self.depth_ms) if self.depth_ms else 0:5.1f}ms"
            f"  p95 {self._p(self.depth_ms,95):5.1f}ms\n"
            f"total  avg {np.mean(self.total_ms) if self.total_ms else 0:5.1f}ms"
            f"  p95 {self._p(self.total_ms,95):5.1f}ms\n"
            f"age    avg {np.mean(self.age_ms) if self.age_ms else 0:5.1f}ms"
            f"  p95 {self._p(self.age_ms,95):5.1f}ms\n"
            f"servo  {moves:.1f} moves/min/servo"
            + self._scene_line_ascii()
        )

    def _scene_line_ascii(self) -> str:
        s = self._scene_share()
        if s is None:
            return ""
        clf = f"{np.mean(self.scene_ms):.1f}ms" if self.scene_ms else "-"
        return (f"\nscene  in {s[0]:.0f}%  out {s[1]:.0f}%  both {s[2]:.0f}%"
                f"  switches {self.n_switches}  clf {clf}")


class Pipeline:
    def __init__(self, cfg: Config, source: Camera | VideoFileSource,
                 link: PicoLink | None = None) -> None:
        self.cfg = cfg
        self.source = source
        self.spec = cfg.depth.spec

        # 깊이 엔진: scene.mode에 따라 실내·실외 중 필요한 것만
        sc = cfg.scene
        paths = {"indoor": cfg.depth.engine, "outdoor": cfg.depth.outdoor_engine}
        self.engines: dict[str, DepthEngine] = {}
        for key in ENGINES_FOR_MODE[sc.mode]:
            p = cfg.path(paths[key])
            if not p.exists():
                flag = " --outdoor" if key == "outdoor" else ""
                raise FileNotFoundError(
                    f"{SCENE_KO[key]} 깊이 엔진 없음: {p}\n"
                    f"  생성: python tools/build_engine.py --spec {cfg.depth.input_spec}{flag}"
                )
            self.engines[key] = DepthEngine(cfg.depth, cfg.grid, p)
        self.depth = next(iter(self.engines.values()))  # 호환용: 첫 엔진
        self._last_models: tuple[str, ...] = (next(iter(self.engines)),)

        # 장면 전환
        self.classifier = None
        self.switcher: SceneSwitcher | None = None
        if sc.mode == "auto":
            self.classifier = SceneClassifier(cfg.path(sc.checkpoint_dir))
        if sc.mode != "off":
            self.switcher = SceneSwitcher(sc, self.classifier)
            logger.info("%s", self.switcher.describe())

        self.mapper = DepthMapper(cfg.mapping, cfg.filter, geo.CELL_COUNT)
        self.hazard = HazardEncoder(cfg.hazard, self.spec.width, self.spec.height)
        self.link = link
        self.stats = Stats()

        self.calib = CalibTable.load_or_default(
            cfg.path(cfg.calibration.file), levels=cfg.mapping.levels
        )
        errs = self.calib.validate()
        for e in errs:
            logger.warning("캘리브레이션: %s", e)

        self._csv_file = None
        self._csv = None
        if cfg.runtime.metrics_csv:
            p = cfg.path(cfg.runtime.metrics_csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = p.open("w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            # scene: 확정 장면(off = 전환 안 함), uncertain: 애매 1/0, p_out: 최근 판별 P(실외),
            # p_smooth: 평활 P(실외), models: 돌린 깊이 모델(indoor|outdoor|both), far_m: 이번 d_far
            self._csv.writerow(
                ["frame", "age_ms", "scene_ms", "depth_ms", "hazard_ms", "filter_ms", "tx_ms",
                 "total_ms", "sent", "scene", "uncertain", "p_out", "p_smooth", "models", "far_m"]
                + [f"d{i}" for i in range(geo.CELL_COUNT)]
                + [f"L{i}" for i in range(geo.CELL_COUNT)]
            )
            logger.info("계측 로그: %s", p)

        self._last_t = time.perf_counter()

    # ── 예열 ──
    def warmup(self, n: int = 10) -> dict[str, float]:
        """
        모든 추론 엔진을 예열한다. 이걸 건너뛰면 첫 프레임에 TRT 엔진
        역직렬화가 실시간 경로로 들어와 초기 지연이 수 초까지 튄다.
        """
        out = {f"depth_{k}_ms": e.warmup(n) for k, e in self.engines.items()}
        out["depth_ms"] = next(iter(out.values()))  # 호환: 첫 엔진
        if self.classifier is not None:
            out["scene_ms"] = self.classifier.warmup(n)
        out["hazard_ms"] = self.hazard.warmup(max(3, n // 2))
        per_frame = max(out[f"depth_{k}_ms"] for k in self.engines) + out["hazard_ms"]
        logger.info(
            "예열 완료: 깊이 %s%s + 위험도 %.1fms → 모델 1개 프레임 약 %.1fms (%.1f fps 상당)",
            " / ".join(f"{SCENE_KO[k]} {out[f'depth_{k}_ms']:.1f}ms" for k in self.engines),
            f", 장면 판별 {out['scene_ms']:.1f}ms(초당 {self.cfg.scene.rate_hz:g}회)"
            if "scene_ms" in out else "",
            out["hazard_ms"], per_frame, 1000 / max(1e-6, per_frame),
        )
        # 예열 중 쌓인 통계는 버린다
        self.stats = Stats()
        self.mapper.reset()
        return out

    # ── 기동 시 Pico 동기화 ──
    def sync_hardware(self) -> None:
        if self.link is None or not self.link.connected:
            return
        t, hz = self.cfg.transport, self.cfg.hazard
        tick_ms = int(round(1000.0 / t.pin_rate_hz))
        jit = [geo.deg_delta_to_ticks(d) for d in hz.jitter_deg]
        self.link.push_config(
            levels=self.cfg.mapping.levels,
            max_simultaneous=self.cfg.safety.max_simultaneous_moves,
            tick_period_ms=tick_ms,
            jitter_ticks=jit,
        )
        logger.info("Pico 박자 %d ms(초당 %.1f회), 위험도 1/2/3 진동 %s도 → %s tick, 진동 초당 %.0f회",
                    tick_ms, t.pin_rate_hz, "/".join(f"{d:g}" for d in hz.jitter_deg),
                    "/".join(str(k) for k in jit), 2 * t.pin_rate_hz)
        self.link.push_calibration(self.calib)
        self.link.send_home()
        d = self.cfg.safety.startup_home_delay_ms / 1000.0
        logger.info("전체 홈 복귀 대기 %.1f초", d)
        time.sleep(d)
        cfg_lines = []
        for line in self.link.drain(max_lines=60):
            if line.startswith(("CFG", "ERR")):
                logger.info("Pico: %s", line)
                if line.startswith("CFG"):
                    cfg_lines.append(line)
            else:
                logger.debug("Pico: %s", line)
        # 펌웨어가 받아들인 값 확인 — 범위 밖 값은 펌웨어가 무시한다. 버전을 모르면 설정을 두 번
        # 보내므로 마지막 CFG 줄로 판단. v2 펌웨어 응답에는 tick=이 없다.
        want = f"tick={tick_ms} jit={','.join(str(k) for k in jit)}"
        if cfg_lines and "tick=" in cfg_lines[-1] and want not in cfg_lines[-1]:
            logger.warning("Pico가 받아들인 설정이 다름: 보낸 값 %s / 응답 %s", want, cfg_lines[-1])
        elif cfg_lines and "tick=" not in cfg_lines[-1]:
            logger.warning("Pico 펌웨어가 박자·진동 각도 설정을 모름(v2) — 받은 값을 바로 적용한다")

    # ── 한 프레임 ──
    def step(self, timeout: float = 1.0) -> FrameResult | None:
        t = StageTimes()
        t0 = time.perf_counter()

        frame = self.source.read(timeout=timeout)
        if frame is None:
            return None
        t.capture_age = (time.perf_counter() - frame.ts) * 1000

        # 장면 판별(초당 rate_hz회) → 이번 프레임에 돌릴 모델과 d_far
        dec: SceneDecision | None = None
        t1 = time.perf_counter()
        if self.switcher is not None:
            dec = self.switcher.step(frame.image, now=frame.ts)
        t.scene = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        models = dec.models if dec is not None else self._last_models
        cells = [self.engines[k].infer(frame.image) for k in models]
        # 두 모델이면 셀마다 더 가까운 값. fmin: 한쪽이 NaN이어도 다른 쪽 값을 쓴다
        cell_dist = cells[0] if len(cells) == 1 else np.fmin.reduce(cells)
        self._last_models = models
        t.depth = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        hazards, dets = self.hazard.encode(frame.image, cell_dist)
        t.hazard = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        if dec is not None:
            self.mapper.set_far(dec.far_m)
        levels = self.mapper.update(cell_dist, dt=dt)
        t.filt = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        sent = False
        if self.link is not None:
            sent = self.link.send_grid(levels, hazards)
        t.tx = (time.perf_counter() - t1) * 1000

        t.total = (time.perf_counter() - t0) * 1000
        r = FrameResult(frame, cell_dist, levels, hazards, dets, t, sent, scene=dec)
        self.stats.update(r)

        if self._csv is not None:
            if dec is None:
                sc = ["off", 0, "", "", models[0]]
            else:
                sc = [dec.state, int(dec.uncertain),
                      "" if dec.p_out is None else round(dec.p_out, 3),
                      "" if dec.p_smooth is None else round(dec.p_smooth, 3),
                      dec.models_label]
            self._csv.writerow(
                [frame.index] + t.as_row() + [int(sent)]
                + sc + [round(self.mapper.active_far_m, 3)]
                + [round(float(v), 3) for v in cell_dist]
                + [int(v) for v in levels]
            )
        return r

    def last_depth_map(self) -> np.ndarray:
        """
        직전 프레임의 전체 깊이맵 (미터). 두 모델을 돌렸으면 픽셀별 가까운 값.
        디버그 시각화용 — D2H 비용 발생.
        """
        maps = [self.engines[k].last_depth_map() for k in self._last_models]
        return maps[0] if len(maps) == 1 else np.fmin.reduce(maps)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        logger.info("파이프라인 통계:\n%s", self.stats.summary())
