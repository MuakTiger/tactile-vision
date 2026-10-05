#!/usr/bin/env bash
# Pico 펌웨어 컴파일 도구 설치 — 젯슨(aarch64), 사용자 영역만 씀(sudo 불필요). 버전 고정, 다시 실행해도 안전.
#
#   arduino-cli 1.5.1                         → ~/.local/bin/arduino-cli (sha256 확인)
#   arduino-pico 코어 rp2040:rp2040@6.2.0      → ~/.arduino15 (툴체인 포함 약 320 MB)
#   Adafruit BusIO 1.17.4,
#   Adafruit PWM Servo Driver Library 3.0.3   → ~/Arduino/libraries
#   arduino:avr 1.8.8 (Uno 코어, 약 250 MB) + Servo 1.3.0   → Uno 캘리브레이션 스케치용
#
# 버전은 2026-10-03/04에 확인한 최신 안정판. 바꿀 때는 아래 변수만 고친다.
# 빌드: tools/build_firmware.sh

set -euo pipefail

ACLI_VER=1.5.1
CORE_VER=6.2.0
BUSIO_VER=1.17.4
PWM_VER=3.0.3
AVR_VER=1.8.8
SERVO_VER=1.3.0
RP2040_INDEX=https://github.com/earlephilhower/arduino-pico/releases/download/global/package_rp2040_index.json

BIN="$HOME/.local/bin"
ACLI="$BIN/arduino-cli"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "이 스크립트는 aarch64(젯슨)용입니다: $(uname -m)" >&2
  exit 1
fi

# ── 1. arduino-cli ──
if [ -x "$ACLI" ] && "$ACLI" version 2>/dev/null | grep -q "Version: $ACLI_VER "; then
  echo "arduino-cli $ACLI_VER 이미 설치됨: $ACLI"
else
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  base="https://github.com/arduino/arduino-cli/releases/download/v$ACLI_VER"
  tgz="arduino-cli_${ACLI_VER}_Linux_ARM64.tar.gz"
  echo "arduino-cli $ACLI_VER 내려받는 중..."
  curl -fsSL -o "$tmp/$tgz" "$base/$tgz"
  curl -fsSL -o "$tmp/checksums.txt" "$base/${ACLI_VER}-checksums.txt"
  # 공식 체크섬 파일과 대조 — 다르면 여기서 멈춘다
  (cd "$tmp" && grep " ${tgz}\$" checksums.txt | sha256sum -c -)
  tar -xzf "$tmp/$tgz" -C "$tmp" arduino-cli
  mkdir -p "$BIN"
  install -m 0755 "$tmp/arduino-cli" "$ACLI"
fi
"$ACLI" version

# ── 2. arduino-pico 코어 (RP2040) ──
# 보드 색인 URL은 전역 설정에 넣지 않고 명령마다 넘긴다.
"$ACLI" core update-index --additional-urls "$RP2040_INDEX"
if "$ACLI" core list --additional-urls "$RP2040_INDEX" | grep -qE "^rp2040:rp2040 +$CORE_VER "; then
  echo "rp2040:rp2040@$CORE_VER 이미 설치됨"
else
  "$ACLI" core install "rp2040:rp2040@$CORE_VER" --additional-urls "$RP2040_INDEX"
fi

# ── 3. 라이브러리 (의존성도 버전 고정: BusIO 먼저, PWM은 --no-deps) ──
"$ACLI" lib update-index
"$ACLI" lib install "Adafruit BusIO@$BUSIO_VER"
"$ACLI" lib install --no-deps "Adafruit PWM Servo Driver Library@$PWM_VER"

# ── 4. Arduino Uno 캘리브레이션 스케치용 (2026-10-04) ──
# 서보 스윕(firmware/calibration/uno_servo_sweep) — 빌드/업로드: tools/upload_uno_sweep.sh
if "$ACLI" core list | grep -qE "^arduino:avr +$AVR_VER "; then
  echo "arduino:avr@$AVR_VER 이미 설치됨"
else
  "$ACLI" core install "arduino:avr@$AVR_VER"
fi
"$ACLI" lib install "Servo@$SERVO_VER"

echo
echo "── 설치 결과 ──"
"$ACLI" core list --additional-urls "$RP2040_INDEX"
"$ACLI" lib list | grep -E "Adafruit (BusIO|PWM Servo)" || true
echo "완료. 빌드: tools/build_firmware.sh"
