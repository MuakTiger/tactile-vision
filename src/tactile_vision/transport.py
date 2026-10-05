"""
Pico 시리얼 전송.

기존 코드의 문제:
  - 매 프레임 reset_input_buffer()/reset_output_buffer() 호출 → 전송 중 패킷 절단.
  - 전송률 제한 없음. 30fps로 쏘면 Pico 수신 버퍼만 채웠다.
  - 연결 실패 시 원인 구분 없음. 실제로는 권한 문제가 대부분이다.

v3 (2026-10-03): Pico 펌웨어가 자기 박자(transport.pin_rate_hz)마다 최신 값 하나만 적용한다.
젯슨은 계산 결과가 나올 때마다 보내고(max_rate_hz는 과전송 방지 상한), 박자·진동 진폭은
기동 시 SET_CONFIG로 알려준다. 펌웨어 버전은 연결 직후 PING 응답이나 기동 배너로 판단한다.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import serial

from . import protocol as P
from .calibration import CalibTable
from .config import TransportCfg

logger = logging.getLogger(__name__)

_READY_RE = re.compile(r"^PICO_READY v(\d+)")
_PONG_V_RE = re.compile(r"\bv=(\d+)")


def parse_fw_version(line: str) -> int | None:
    """Pico 상태 줄 → 펌웨어 버전. 'PICO_READY vN …' 또는 PONG(v3부터 'v=N', v2에는 없음)."""
    line = line.strip()
    m = _READY_RE.match(line)
    if m:
        return int(m.group(1))
    if line.startswith("PONG"):
        m = _PONG_V_RE.search(line)
        return int(m.group(1)) if m else 2
    return None


def diagnose_port(port: str) -> str:
    """연결 실패 원인을 구체적으로 짚어준다."""
    p = Path(port)
    if not p.exists():
        globbed = sorted(Path("/dev").glob("ttyACM*")) + sorted(
            Path("/dev").glob("ttyUSB*")
        )
        if globbed:
            return f"{port} 없음. 발견된 포트: {[str(g) for g in globbed]}"
        return (
            f"{port} 없음. Pico가 연결되지 않았거나 BOOTSEL 모드입니다. "
            "lsusb로 확인하세요."
        )
    if not os.access(port, os.R_OK | os.W_OK):
        import grp

        try:
            gid = p.stat().st_gid
            gname = grp.getgrgid(gid).gr_name
        except (KeyError, OSError):
            gname = "?"
        user = os.environ.get("USER", "현재 사용자")
        return (
            f"{port} 접근 권한 없음 (그룹 '{gname}'). 해결:\n"
            f"    sudo usermod -aG {gname} {user}\n"
            f"    이후 재로그인(또는 newgrp {gname}) 필요"
        )
    return f"{port} 존재하고 권한도 있으나 열기 실패 — 다른 프로세스가 점유 중일 수 있음"


class PicoLink:
    """
    Pico 연결. 하드웨어가 없어도 동작한다(offline 모드)면 파이프라인 개발이 막히지
    않는다. connected 프로퍼티로 실제 연결 여부를 확인할 수 있다.
    """

    def __init__(self, cfg: TransportCfg, required: bool = False) -> None:
        self.cfg = cfg
        self.conn: serial.Serial | None = None
        self.fw_version: int | None = None  # 연결 후 판단. None = 응답 없음
        self._seq = 0
        self._last_tx = 0.0
        self._min_period = 1.0 / cfg.max_rate_hz if cfg.max_rate_hz > 0 else 0.0
        self._lock = threading.Lock()  # 캘리브레이션 도구는 keepalive 스레드와 같이 보낸다

        # 통계
        self.n_sent = 0
        self.n_skipped = 0  # 전송 상한으로 건너뜀
        self.n_error = 0
        self.rx_lines: list[str] = []

        try:
            self.conn = serial.Serial(
                cfg.port, cfg.baud, timeout=0.05, write_timeout=0.2
            )
        except (serial.SerialException, OSError) as e:
            msg = diagnose_port(cfg.port)
            if required:
                raise RuntimeError(f"Pico 연결 실패: {e}\n  {msg}") from e
            logger.warning("Pico 연결 실패 → 오프라인 모드로 계속\n  %s", msg)
            return

        # arduino-pico는 포트를 열어도 재부팅하지 않아 기동 배너(PICO_READY)를 놓치기 쉽다.
        # PING을 먼저 보내고 배너나 PONG 중 먼저 오는 줄로 펌웨어 버전을 판단한다.
        self.send_ping()
        deadline = time.time() + 3.0
        first = ""
        while time.time() < deadline:
            line = self.conn.readline().decode("utf-8", "ignore").strip()
            if not line:
                continue
            self.rx_lines.append(line)
            v = parse_fw_version(line)
            if v is not None:
                self.fw_version, first = v, line
                break
        if self.fw_version is None:
            logger.warning("Pico 응답(PICO_READY/PONG) 없음 — 펌웨어가 없거나 부팅 중일 수 있음. 계속합니다.")
        elif self.fw_version < 3:
            logger.warning(
                "Pico 펌웨어 v%d — 박자 적용·진동 각도 설정은 v3부터. 지금은 받은 값을 바로 적용한다. "
                "firmware/pico/tactile_pico 업로드 필요 (tools/build_firmware.sh)", self.fw_version)
        else:
            logger.info("Pico 연결: %s 펌웨어 v%d (%s)", cfg.port, self.fw_version, first)

    @property
    def connected(self) -> bool:
        return self.conn is not None and self.conn.is_open

    # ── 송신 ──
    def _send(self, make: Callable[[int], bytes]) -> bool:
        """시퀀스 번호를 붙여 한 프레임 전송. 여러 스레드에서 불러도 순서·번호가 섞이지 않는다."""
        if not self.connected:
            return False
        with self._lock:
            self._seq = (self._seq + 1) & 0xFF
            frame = make(self._seq)
            try:
                self.conn.write(frame)
                self.n_sent += 1
                return True
            except (serial.SerialException, serial.SerialTimeoutException, OSError) as e:
                self.n_error += 1
                if self.n_error % 20 == 1:
                    logger.error("전송 실패(%d회): %s", self.n_error, e)
                return False

    def send_grid(
        self,
        levels: np.ndarray,
        hazards: np.ndarray | None = None,
        estop: bool = False,
        force: bool = False,
    ) -> bool:
        """
        계산 결과가 나올 때마다 부른다. Pico(v3)가 박자(pin_rate_hz)마다 최신 값 하나만 적용하므로
        여기서 박자를 맞출 필요는 없다. max_rate_hz는 과전송 방지 상한일 뿐이다.
        """
        now = time.perf_counter()
        if not force and self._min_period and (now - self._last_tx) < self._min_period:
            self.n_skipped += 1
            return False
        self._last_tx = now
        flags = P.FLAG_ESTOP if estop else P.FLAG_NONE
        return self._send(lambda s: P.grid_frame(levels, hazards, s, flags))

    def send_raw_tick(self, channel: int, tick: int) -> bool:
        return self._send(lambda s: P.raw_tick_frame(channel, tick, s))

    def send_home(self) -> bool:
        return self._send(P.home_all_frame)

    def send_ping(self) -> bool:
        return self._send(P.ping_frame)

    def send_keepalive(self) -> bool:
        """failsafe만 막는 빈 프레임. 핀 값을 보내지 않는 동안(캘리브레이션)에만 쓴다."""
        return self._send(P.keepalive_frame)

    def push_calibration(self, table: CalibTable) -> int:
        """캘리브레이션 테이블 전체를 Pico에 반영. 반영된 채널 수 반환."""
        if not self.connected:
            logger.warning("오프라인 — 캘리브레이션 전송 생략")
            return 0
        n = 0
        for c in sorted(table.channels, key=lambda c: c.ch):
            if self._send(lambda s, c=c: P.set_limit_frame(c.ch, c.home_tick, c.max_tick, s)):
                n += 1
            time.sleep(0.005)  # Pico가 각 프레임을 처리할 여유
        logger.info("캘리브레이션 %d/%d 채널 전송", n, len(table.channels))
        return n

    def push_config(self, levels: int, max_simultaneous: int, tick_period_ms: int,
                    jitter_ticks: Sequence[int]) -> bool:
        """
        펌웨어 버전에 맞는 SET_CONFIG. v3: 박자·진동 진폭 포함 7바이트.
        v2: 4바이트(levels·동시 구동만 의미 있음). 버전 모름: 4바이트 뒤 7바이트
        (v3는 둘 다 받고, v2는 7바이트를 길이 오류로 무시한다).
        """
        v = self.fw_version
        if v is not None and v >= 3:
            return self._send(lambda s: P.config_frame(levels, max_simultaneous, tick_period_ms,
                                                       jitter_ticks, s))
        ok = self._send(lambda s: P.config_frame_legacy(levels, max_simultaneous, seq=s))
        if v is None:
            ok = self._send(lambda s: P.config_frame(levels, max_simultaneous, tick_period_ms,
                                                     jitter_ticks, s)) and ok
        return ok

    # ── 수신 (저빈도 ASCII) ──
    def drain(self, max_lines: int = 20) -> list[str]:
        """Pico가 보낸 상태 메시지를 읽어온다. 블로킹하지 않는다."""
        if not self.connected:
            return []
        out = []
        try:
            while self.conn.in_waiting and len(out) < max_lines:
                line = self.conn.readline().decode("utf-8", "ignore").strip()
                if line:
                    out.append(line)
        except (serial.SerialException, OSError):
            pass
        self.rx_lines.extend(out)
        del self.rx_lines[:-200]  # 최근 200줄만 보관
        return out

    def close(self, home_first: bool = True) -> None:
        if not self.connected:
            return
        if home_first:
            self.send_home()
            time.sleep(0.1)
        self.conn.close()
        logger.info(
            "Pico 종료: 전송 %d / 상한스킵 %d / 오류 %d",
            self.n_sent, self.n_skipped, self.n_error,
        )

    def __enter__(self) -> PicoLink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
