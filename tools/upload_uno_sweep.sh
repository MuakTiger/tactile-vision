#!/usr/bin/env bash
# Uno 서보 스윕 스케치(firmware/calibration/uno_servo_sweep) 컴파일 + 업로드
#
#   tools/upload_uno_sweep.sh [포트]        포트 기본값 /dev/ttyUSB0 (FTDI 호환 보드) — 정품 Uno는 /dev/ttyACM*
#   tools/upload_uno_sweep.sh --compile     컴파일만
#
# 시리얼 모니터(115200): ? 로 도움말. 도구 설치: tools/setup_firmware_toolchain.sh (arduino:avr 1.8.8, Servo 1.3.0)
# 로그인 그룹(dialout)이 아직 적용 전이면 sg dialout 으로 자동 재실행한다.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ACLI="${ARDUINO_CLI:-$HOME/.local/bin/arduino-cli}"
FQBN="arduino:avr:uno"
SKETCH="$REPO/firmware/calibration/uno_servo_sweep"

if [ ! -x "$ACLI" ]; then
  echo "arduino-cli 없음: $ACLI — 먼저 tools/setup_firmware_toolchain.sh" >&2
  exit 1
fi

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
"$ACLI" compile --fqbn "$FQBN" --warnings all --output-dir "$OUT" "$SKETCH" | grep -E 'Sketch uses|Global variables'

[ "${1:-}" = "--compile" ] && exit 0

PORT="${1:-/dev/ttyUSB0}"
if [ ! -e "$PORT" ]; then
  echo "포트 없음: $PORT — 보드를 꽂았는지, ls /dev/ttyUSB* /dev/ttyACM* 로 확인" >&2
  exit 1
fi
if [ ! -w "$PORT" ]; then
  echo "$PORT 쓰기 권한 없음 → sg dialout 으로 다시 실행합니다"
  exec sg dialout -c "'$0' '$PORT'"
fi
"$ACLI" upload --fqbn "$FQBN" -p "$PORT" --input-dir "$OUT" --verify "$SKETCH"
echo "업로드 완료: $PORT"
