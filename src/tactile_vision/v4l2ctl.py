"""
V4L2 카메라 컨트롤 (v4l-utils 없이 ioctl 직접).

자동초점 카메라(Logitech C920 등)의 자동초점을 끄고 초점값을 고정하는 데 쓴다.
자동초점이 렌즈를 움직이면 초점거리(화각)도 조금씩 바뀌어 metric 거리 추정과
카메라 캘리브레이션이 흔들리기 때문이다.

    with V4L2Ctrl("/dev/video0") as ctrl:
        focus = Focus(ctrl)
        focus.lock(0)          # 자동초점 끔 + 초점값 0 고정

C920 (2026-10-02 확인): 초점값 0~250, 5 단위, 0 = 무한대 쪽, 클수록 가까운 쪽.
설정은 카메라를 뽑을 때까지 유지되고, 다시 꽂으면 자동초점이 켜진 기본값으로 돌아간다.
그래서 카메라를 열 때마다 적용해야 한다. 캡처와 별도 fd를 쓰므로 스트리밍 중에도 바꿀 수 있다.
"""

from __future__ import annotations

import fcntl
import logging
import os
import struct

logger = logging.getLogger(__name__)


def _iowr(nr: int, size: int) -> int:
    """_IOWR('V', nr, size)"""
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


_QFMT = "=II32siiiiI2I"  # struct v4l2_queryctrl (68 B)
_CFMT = "=Ii"  # struct v4l2_control (8 B)
VIDIOC_QUERYCTRL = _iowr(36, struct.calcsize(_QFMT))
VIDIOC_G_CTRL = _iowr(27, struct.calcsize(_CFMT))
VIDIOC_S_CTRL = _iowr(28, struct.calcsize(_CFMT))
CID_FOCUS_ABSOLUTE = 0x009A090A
CID_FOCUS_AUTO = 0x009A090C
CTRL_FLAG_DISABLED = 0x0001


def video_node(device: int | str) -> str | None:
    """설정의 camera.device → /dev/videoN 경로. V4L2 장치 번호·경로가 아니면 None."""
    if isinstance(device, int):
        return f"/dev/video{device}"
    if isinstance(device, str):
        if device.isdigit():
            return f"/dev/video{int(device)}"
        if device.startswith("/dev/video"):
            return device
    return None


class V4L2Ctrl:
    """장치 노드의 컨트롤 읽기/쓰기. 노드를 못 열면 '컨트롤 없음'처럼 동작한다."""

    def __init__(self, dev: str) -> None:
        self.dev = dev
        try:
            self.fd: int | None = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
        except OSError as e:
            logger.warning("컨트롤용으로 %s 열기 실패: %s", dev, e)
            self.fd = None

    def query(self, cid: int) -> tuple[int, int, int] | None:
        """(최소, 최대, 단계). 컨트롤이 없으면 None."""
        if self.fd is None:
            return None
        buf = bytearray(struct.pack(_QFMT, cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(self.fd, VIDIOC_QUERYCTRL, buf)
        except OSError:
            return None
        _, _, _, lo, hi, step, _, flags, _, _ = struct.unpack(_QFMT, buf)
        return None if flags & CTRL_FLAG_DISABLED else (lo, hi, max(1, step))

    def get(self, cid: int) -> int | None:
        if self.fd is None:
            return None
        buf = bytearray(struct.pack(_CFMT, cid, 0))
        try:
            fcntl.ioctl(self.fd, VIDIOC_G_CTRL, buf)
        except OSError:
            return None
        return struct.unpack(_CFMT, buf)[1]

    def set(self, cid: int, value: int) -> bool:
        if self.fd is None:
            return False
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_CTRL, bytearray(struct.pack(_CFMT, cid, int(value))))
        except OSError as e:
            logger.warning("카메라 컨트롤 0x%08X ← %d 실패 (%s): %s", cid, value, self.dev, e)
            return False
        return True

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> V4L2Ctrl:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Focus:
    """자동초점·초점값. 카메라에 해당 컨트롤이 없으면(고정초점) 아무것도 하지 않는다."""

    def __init__(self, ctrl: V4L2Ctrl) -> None:
        self.ctrl = ctrl
        self.has_auto = ctrl.query(CID_FOCUS_AUTO) is not None
        self.range = ctrl.query(CID_FOCUS_ABSOLUTE)  # (최소, 최대, 단계) | None

    @property
    def controllable(self) -> bool:
        return self.has_auto or self.range is not None

    def auto(self) -> bool:
        return self.has_auto and bool(self.ctrl.get(CID_FOCUS_AUTO))

    def value(self) -> int | None:
        return self.ctrl.get(CID_FOCUS_ABSOLUTE) if self.range else None

    def snap(self, v: int) -> int:
        """범위·단계에 맞춘 값."""
        if self.range is None:
            return v
        lo, hi, step = self.range
        return min(hi, max(lo, lo + round((v - lo) / step) * step))

    def set_auto(self, on: bool) -> None:
        if self.has_auto:
            self.ctrl.set(CID_FOCUS_AUTO, int(on))

    def set_value(self, v: int) -> None:
        """초점값 설정. 자동초점이 켜져 있으면 카메라가 무시한다."""
        if self.range is not None:
            self.ctrl.set(CID_FOCUS_ABSOLUTE, self.snap(v))

    def lock(self, v: int) -> None:
        """
        자동초점을 끄고 초점값 v로 고정.
        자동초점이 렌즈를 옮겨 놓은 뒤에는 보고값과 실제 렌즈 위치가 다를 수 있어서,
        이웃 값을 먼저 써서 렌즈가 실제로 v까지 움직이게 한다.
        """
        self.set_auto(False)
        if self.range is None:
            return
        _, hi, step = self.range
        target = self.snap(v)
        self.set_value(target + step if target + step <= hi else target - step)
        self.set_value(target)

    def label(self) -> str:
        if not self.controllable:
            return "fixed-focus camera"
        return "AF on" if self.auto() else f"AF off, focus {self.value()}"
