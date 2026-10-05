#!/usr/bin/env bash
# Pico 펌웨어 컴파일 → firmware/pico/build/tactile_pico.ino.uf2
#
#   tools/build_firmware.sh                       컴파일만
#   tools/build_firmware.sh --upload [/dev/ttyACM0]   컴파일 후 arduino-cli로 업로드
#                                                 (june이 dialout 그룹이어야 함)
#
# 업로드를 arduino-cli 없이 하려면(처음 올릴 때 권장): Pico의 BOOTSEL 버튼을 누른 채 USB를 꽂으면
# RPI-RP2 저장장치가 뜬다 → build/tactile_pico.ino.uf2를 거기에 복사하면 자동으로 재부팅된다.
#
# 도구 설치: tools/setup_firmware_toolchain.sh

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ACLI="${ARDUINO_CLI:-$HOME/.local/bin/arduino-cli}"
FQBN="rp2040:rp2040:rpipico"
SKETCH="$REPO/firmware/pico/tactile_pico"
OUT="$REPO/firmware/pico/build"

if [ ! -x "$ACLI" ]; then
  echo "arduino-cli 없음: $ACLI — 먼저 tools/setup_firmware_toolchain.sh" >&2
  exit 1
fi

# PC 하네스와 같은 핵심 코드인지 — 빌드 전에 PC에서 같은 헤더로 시험하려면: python tools/test_firmware.py
"$ACLI" compile --fqbn "$FQBN" --warnings all --output-dir "$OUT" "$SKETCH"
echo
ls -la "$OUT"/*.uf2

if [ "${1:-}" = "--upload" ]; then
  PORT="${2:-/dev/ttyACM0}"
  "$ACLI" upload --fqbn "$FQBN" -p "$PORT" --input-dir "$OUT" "$SKETCH"
  echo "업로드 완료: $PORT"
fi
