"""
카메라 캡처.

기존 코드의 문제:
  - cap.read()를 추론과 같은 스레드에서 호출해 루프가 느려지면 V4L2 내부 큐에
    프레임이 쌓이고, 읽는 프레임은 계속 과거가 된다(지연 누적).
  - BUFFERSIZE 미설정.
  - 1280×720을 518×518로 강제 리사이즈해 가로를 44% 압축.

해결:
  - 전용 스레드가 계속 읽어 최신 프레임 1장만 보관(오래된 프레임 폐기).
  - 목표 종횡비로 센터 크롭 후 리사이즈 → 기하 왜곡 제거.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraCfg
from .v4l2ctl import Focus, V4L2Ctrl, video_node

logger = logging.getLogger(__name__)


def crop_rect(src_w: int, src_h: int, target_aspect: float) -> tuple[int, int, int, int]:
    """
    센터 크롭 사각형 (x, y, w, h). 목표 종횡비에 맞춰 최소한만 잘라낸다.
    원본보다 넓은 종횡비를 원하면 위아래를, 좁으면 좌우를 자른다.
    """
    src_aspect = src_w / src_h
    if abs(src_aspect - target_aspect) < 1e-6:
        return 0, 0, src_w, src_h
    if src_aspect > target_aspect:
        # 원본이 더 넓음 → 좌우를 자른다
        w = int(round(src_h * target_aspect))
        w = min(w, src_w)
        return (src_w - w) // 2, 0, w, src_h
    # 원본이 더 좁음 → 위아래를 자른다
    h = int(round(src_w / target_aspect))
    h = min(h, src_h)
    return 0, (src_h - h) // 2, src_w, h


@dataclass
class Frame:
    image: np.ndarray  # BGR, 크롭+리사이즈 완료
    raw: np.ndarray  # 원본 BGR (디버그 시각화용)
    ts: float  # 캡처 시각 (perf_counter)
    index: int


class Camera:
    """
    최신 프레임만 유지하는 스레드 캡처.

    with Camera(cfg, 504, 280) as cam:
        frame = cam.read(timeout=1.0)
    """

    def __init__(self, cfg: CameraCfg, out_w: int, out_h: int) -> None:
        self.cfg = cfg
        self.out_w = out_w
        self.out_h = out_h
        self.target_aspect = out_w / out_h

        self._cap: cv2.VideoCapture | None = None
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._crop: tuple[int, int, int, int] | None = None

        # 통계
        self.n_captured = 0
        self.n_dropped = 0
        self.n_read_fail = 0

    # ── 수명 관리 ──
    def open(self) -> None:
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(
                f"카메라를 열 수 없음: device={self.cfg.device}. "
                "연결과 /dev/video* 존재를 확인할 것."
            )
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # 드라이버 큐를 최소화해 지연 누적을 막는다
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._apply_focus()

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "카메라 열림: 요청 %dx%d@%d → 실제 %dx%d@%.0f (%s)",
            self.cfg.width, self.cfg.height, self.cfg.fps, aw, ah, afps, self.cfg.fourcc,
        )
        if (aw, ah) != (self.cfg.width, self.cfg.height):
            logger.warning("카메라가 요청 해상도를 거부함 — 실제값으로 진행")

        self._crop = crop_rect(aw, ah, self.target_aspect)
        x, y, w, h = self._crop
        logger.info(
            "크롭 %dx%d+%d+%d (종횡비 %.3f) → 리사이즈 %dx%d | 버려지는 화각 %.1f%%",
            w, h, x, y, w / h, self.out_w, self.out_h,
            100 * (1 - (w * h) / (aw * ah)),
        )
        self._cap = cap

    def _apply_focus(self) -> None:
        """
        camera.autofocus / camera.focus 적용. C920은 다시 꽂으면 자동초점이 켜지므로
        열 때마다 적용한다. 초점 컨트롤이 없는 카메라(고정초점)는 그대로 둔다.
        """
        af, fv = self.cfg.autofocus, self.cfg.focus
        if af is None and fv is None:
            return
        dev = video_node(self.cfg.device)
        if dev is None:
            logger.info("초점 설정 건너뜀: V4L2 장치가 아님 (device=%s)", self.cfg.device)
            return
        with V4L2Ctrl(dev) as ctrl:
            focus = Focus(ctrl)
            if not focus.controllable:
                logger.info("초점: 컨트롤 없음 — 고정초점 카메라, 설정 무시")
                return
            if af is True:
                focus.set_auto(True)
            elif fv is not None:  # af False 또는 None + 초점값 지정 → 고정
                focus.lock(fv)
            else:
                focus.set_auto(False)  # 끄기만, 렌즈는 그 자리
            logger.info("초점: %s", focus.label())
            if fv is not None and af is not True and focus.value() != focus.snap(fv):
                logger.warning("초점값이 설정과 다름: 설정 %d, 카메라 %s", fv, focus.value())

    def start(self) -> None:
        if self._cap is None:
            self.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None and self._crop is not None
        x, y, w, h = self._crop
        idx = 0
        while not self._stop.is_set():
            ok, raw = self._cap.read()
            if not ok:
                self.n_read_fail += 1
                if self.n_read_fail % 30 == 1:
                    logger.warning("프레임 읽기 실패 누적 %d회", self.n_read_fail)
                time.sleep(0.01)
                continue
            ts = time.perf_counter()
            roi = raw[y : y + h, x : x + w]
            # 축소는 INTER_AREA가 앨리어싱이 적다
            img = cv2.resize(roi, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            if self.cfg.flip_horizontal:
                img = cv2.flip(img, 1)
                raw = cv2.flip(raw, 1)
            idx += 1
            self.n_captured += 1
            with self._new:
                if self._latest is not None:
                    self.n_dropped += 1  # 소비되지 않은 프레임을 버림
                self._latest = Frame(image=img, raw=raw, ts=ts, index=idx)
                self._new.notify()

    def read(self, timeout: float = 1.0) -> Frame | None:
        """가장 최신 프레임을 소비한다. 없으면 timeout까지 대기."""
        with self._new:
            if self._latest is None:
                self._new.wait(timeout)
            f, self._latest = self._latest, None
            return f

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info(
            "카메라 종료: 캡처 %d / 폐기 %d / 읽기실패 %d",
            self.n_captured, self.n_dropped, self.n_read_fail,
        )

    def __enter__(self) -> Camera:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoFileSource:
    """
    동영상/이미지 파일 소스. 카메라 없이 재현 가능한 테스트를 하기 위한 대체 경로.
    실험 데이터를 녹화해두고 매핑 파라미터를 바꿔가며 비교할 때 쓴다.
    """

    def __init__(self, path: str, out_w: int, out_h: int, loop: bool = True) -> None:
        self.path = path
        self.out_w, self.out_h = out_w, out_h
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"파일을 열 수 없음: {path}")
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._crop = crop_rect(aw, ah, out_w / out_h)
        self.n_captured = 0
        self.n_dropped = 0
        self.n_read_fail = 0
        logger.info("파일 소스: %s (%dx%d)", path, aw, ah)

    def read(self, timeout: float = 1.0) -> Frame | None:
        ok, raw = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, raw = self.cap.read()
            if not ok:
                return None
        x, y, w, h = self._crop
        img = cv2.resize(
            raw[y : y + h, x : x + w], (self.out_w, self.out_h),
            interpolation=cv2.INTER_AREA,
        )
        self.n_captured += 1
        return Frame(image=img, raw=raw, ts=time.perf_counter(), index=self.n_captured)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoFileSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
