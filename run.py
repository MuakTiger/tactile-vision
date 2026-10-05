#!/usr/bin/env python3
"""
시각-촉각 감각대체 시스템 — 메인 실행기.

  python run.py                      카메라 + 하드웨어 (실내/실외 자동 전환)
  python run.py --debug              시각화 창 포함
  python run.py --no-hw              Pico 없이 (파이프라인만)
  python run.py --source clip.mp4    녹화 영상으로 재현 실험
  python run.py --scene indoor       실내 모델·d_far로 고정 (outdoor / off / auto)
  python run.py --show-mapping       매핑 표만 출력하고 종료
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tactile_vision import geometry as geo  # noqa: E402
from tactile_vision.camera import Camera, VideoFileSource  # noqa: E402
from tactile_vision.config import load  # noqa: E402
from tactile_vision.mapping import DepthMapper  # noqa: E402
from tactile_vision.pipeline import Pipeline  # noqa: E402
from tactile_vision.switching import KO as SCENE_KO  # noqa: E402
from tactile_vision.transport import PicoLink  # noqa: E402

log = logging.getLogger("run")
_stop = False


def _mapping_tables(cfg) -> list[tuple[str, float | None]]:
    """(제목, d_far) 목록. scene.mode에 따라 장면별 d_far 표를 보여준다. None = mapping.far_m."""
    s = cfg.scene
    if s.mode == "off":
        return [("", None)]
    far = {"indoor": s.far_indoor_m, "outdoor": s.far_outdoor_m}
    keys = ("indoor", "outdoor") if s.mode == "auto" else (s.mode,)
    return [(f"[{SCENE_KO[k]}] ", far[k]) for k in keys]


def _describe_mapping(cfg) -> str:
    out = []
    for title, far in _mapping_tables(cfg):
        mp = DepthMapper(cfg.mapping, cfg.filter, geo.CELL_COUNT)
        mp.set_far(far)
        out.append(title + mp.describe_edges())
    return "\n".join(out)


def _on_signal(signum: int, _frame: object) -> None:
    global _stop
    _stop = True
    log.info("종료 신호 수신 (%s) — 핀을 내리고 종료합니다", signum)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None, help="설정 파일 (기본 config/default.yaml)")
    p.add_argument("--source", default=None, help="영상 파일 경로 (미지정 시 카메라)")
    p.add_argument("--no-hw", action="store_true", help="Pico 없이 실행")
    p.add_argument("--debug", action="store_true", help="시각화 창 표시")
    p.add_argument("--port", default=None, help="Pico 포트 (설정 override)")
    p.add_argument("--mapping", default=None,
                   choices=["linear", "log", "disparity", "relative_ema"],
                   help="매핑 모드 override — 실험 조건 비교용")
    p.add_argument("--levels", type=int, default=None, help="단계 수 override")
    p.add_argument("--scene", default=None, choices=["auto", "indoor", "outdoor", "off"],
                   help="장면 전환 override — auto 자동 / indoor·outdoor 고정 / off 예전 방식")
    p.add_argument("--duration", type=float, default=0.0,
                   help="지정 초 후 자동 종료 (0=무한)")
    p.add_argument("--show-mapping", action="store_true",
                   help="매핑 단계 표를 출력하고 종료")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load(args.config)

    # CLI override
    if args.port:
        cfg.transport.port = args.port
    if args.mapping:
        cfg.mapping.mode = args.mapping
    if args.levels:
        cfg.mapping.levels = args.levels
    if args.scene:
        cfg.scene.mode = args.scene

    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    warns = cfg.validate()
    for w in warns:
        log.warning("설정: %s", w)

    if args.show_mapping:
        print(geo.summary())
        print()
        print(_describe_mapping(cfg))
        return 0

    print(geo.summary())
    print()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    spec = cfg.depth.spec
    source = (
        VideoFileSource(args.source, spec.width, spec.height)
        if args.source
        else Camera(cfg.camera, spec.width, spec.height)
    )
    link = None if args.no_hw else PicoLink(cfg.transport)

    try:
        if isinstance(source, Camera):
            source.start()
        pipe = Pipeline(cfg, source, link)
    except Exception as e:  # noqa: BLE001
        log.error("초기화 실패: %s", e)
        source.close()
        if link:
            link.close(home_first=False)
        return 1

    log.info("예열 중 (깊이%s + 위험도 엔진)...",
             " 실내·실외 + 장면 판별" if cfg.scene.mode == "auto" else "")
    pipe.warmup(10)
    pipe.sync_hardware()

    log.info("매핑:\n%s", _describe_mapping(cfg))
    log.info("시작 — 종료는 Ctrl+C%s", " 또는 창에서 q" if args.debug else "")

    if args.debug:
        import cv2

        from tactile_vision import viz

        cv2.namedWindow("tactile-vision", cv2.WINDOW_NORMAL)

    t_end = time.perf_counter() + args.duration if args.duration > 0 else None
    last_report = time.perf_counter()
    last_drain = last_ping = last_report
    placed = False

    try:
        while not _stop:
            r = pipe.step(timeout=1.0)
            if r is None:
                if args.source:
                    log.info("영상 끝")
                    break
                continue

            if args.debug:
                dm = pipe.last_depth_map()
                canvas = viz.compose(r, dm, cfg, pipe.stats.summary_ascii())
                cv2.imshow("tactile-vision", canvas)
                if not placed:  # 첫 프레임: 창을 화면 크기(1:1)에 맞추고 왼쪽 위로
                    h, w = canvas.shape[:2]
                    cv2.resizeWindow("tactile-vision", w, h)
                    cv2.moveWindow("tactile-vision", 20, 40)
                    placed = True
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # 2 s마다 상태 한 줄 — 창 모드에서도 로그로 남긴다
            now = time.perf_counter()
            if now - last_report >= 2.0:
                last_report = now
                lv = r.levels.reshape(geo.GRID_ROWS, geo.GRID_COLS)
                rows = " | ".join(
                    "".join(str(v) for v in row) for row in lv
                )
                sc = ""
                if r.scene is not None:
                    s = r.scene
                    p = "-" if s.p_smooth is None else f"{s.p_smooth * 100:.0f}%"
                    sc = (f"  장면 {SCENE_KO[s.state]}{'(애매)' if s.uncertain else ''}"
                          f" P(실외) {p} 모델 {SCENE_KO[s.models_label]}"
                          f" d_far {s.far_m:.1f}m")
                log.info(
                    "%.1f fps  지연 %.0fms  레벨[%s]  전송 %d%s",
                    1000 / max(r.times.total, 1e-6), r.times.total,
                    rows, pipe.stats.n_sent, sc,
                )

            # Pico 상태 줄은 두 모드 모두 2 s마다 읽는다(안 읽으면 Pico 송신 버퍼가 찬다).
            # 10 s마다 PING → PONG: 박자 수·적용 수·덮어써진 수(sup)로 박자 동작을 확인한다.
            if link and link.connected:
                now = time.perf_counter()
                if now - last_drain >= 2.0:
                    last_drain = now
                    if now - last_ping >= 10.0:
                        last_ping = now
                        link.send_ping()
                    for line in link.drain():
                        log.info("Pico: %s", line)

            if t_end and time.perf_counter() > t_end:
                log.info("지정 시간 경과 — 종료")
                break
    finally:
        if args.debug:
            import cv2

            cv2.destroyAllWindows()
        pipe.close()
        source.close()
        if link:
            link.close(home_first=True)

    print()
    print(pipe.stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
