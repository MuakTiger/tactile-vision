/**
 * 시각-촉각 감각대체 장치 — Raspberry Pi Pico 펌웨어 v3
 *
 * Jetson --USB CDC--> Pico --I2C 400 kHz--> PCA9685 ×2 --> SG90 ×24
 *
 * 동작 규칙은 전부 tactile_core.h에 있다(PC 시험 하네스와 같은 코드). 이 파일은 하드웨어 연결만:
 *   PCA9685 쓰기, USB 시리얼 입출력, 기동 시 시간차 홈 복귀.
 *
 * v2 → v3 (2026-10-03)
 *   · 받은 GRID를 바로 적용하지 않고 보관했다가 박자(기본 200 ms = 초당 5회)마다 최신 값만 적용
 *   · 위험도 진동을 박자에 맞춤: 박자의 2·4번째 1/4 구간, 위험도 1/2/3 → 기본 2/4/6 tick(≈1/2/3도)
 *   · 맨 아래(홈)에 있는 핀은 위로 진동, 보정 범위 밖으로 안 나감
 *   · KEEPALIVE(0x07), SET_CONFIG 7바이트(박자·진동 진폭) — 4바이트 v2 형식도 받음
 *   · 상태 줄은 USB 송신 버퍼가 차 있으면 버린다(젯슨이 안 읽을 때 루프가 최대 1 s 멈추는 것 방지)
 *   v2 원본: ../legacy/tactile_pico_v2/tactile_pico_v2.ino
 *
 * 배선: GP10=SDA, GP7=SCL / PCA9685 0x40(채널 0~11), 0x41(채널 12~23, A0 납땜)
 *       외부 5 V → PCA9685 V+ (서보 전원, Pico 3V3와 분리)
 *
 * 2026-10-05: GP3=SDA/GP7=SCL로 배선됨 — GP3·GP7 모두 SCL 전용이라 I2C가 성립하지 않아
 * GP10=SDA, GP7=SCL(둘 다 I2C1 → Wire1)로 변경. GP3 쪽 선을 GP10으로 옮길 것 (GP6은 GP7과 가까워 피함).
 * PING을 받을 때마다 "I2C sda=GP10 scl=GP7 0x40=ok 0x41=ok map=colpairs" 줄로 두 보드 응답을 보고한다(배선 점검용).
 * 칸 → 채널 (2026-10-05): 보드마다 왼쪽 위부터 세로 2칸씩 — tactile_core.h CELL_TO_CHANNEL.
 *   0x40: 1행 채널 0 2 4 6 8 10 / 2행 1 3 5 7 9 11      0x41: 3행·4행 같은 배열
 *
 * 빌드: tools/build_firmware.sh   (도구 설치: tools/setup_firmware_toolchain.sh)
 * 의존: arduino-pico 코어 rp2040:rp2040@6.2.0, Adafruit PWM Servo Driver Library 3.0.3, Adafruit BusIO 1.17.4
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#include "tactile_core.h"

static constexpr uint8_t  ADDR_LOW  = 0x40;
static constexpr uint8_t  ADDR_HIGH = 0x41;
static constexpr uint32_t I2C_HZ    = 400000;
static constexpr uint16_t PWM_FREQ  = 50;
static constexpr int      RX_BUDGET = 512;  // loop 1회에 처리할 최대 수신 바이트 — 서보 갱신 주기를 지킨다

// I2C 핀 — RP2040은 짝수 GPIO가 SDA, 홀수가 SCL이고 4개마다 I2C0/I2C1이 바뀐다.
// GP10(SDA)·GP7(SCL)은 둘 다 I2C1 → arduino-pico에서는 Wire1이어야 한다.
// (Wire = I2C0에 이 핀을 주면 setSDA()가 panic으로 펌웨어를 멈춘다)
static constexpr uint8_t  PIN_SDA = 10;
static constexpr uint8_t  PIN_SCL = 7;
#define I2C_BUS Wire1

Adafruit_PWMServoDriver pwmLow(ADDR_LOW, I2C_BUS);
Adafruit_PWMServoDriver pwmHigh(ADDR_HIGH, I2C_BUS);
tv::Core core;

// 칸 → 보드·채널: tactile_core.h CELL_TO_CHANNEL (보드마다 왼쪽 위부터 세로 2칸씩)
static void writePwm(uint8_t cell, uint16_t tick) {
  const uint8_t ch = tv::CELL_TO_CHANNEL[cell];
  if (tv::boardOf(cell) == 0) pwmLow.setPWM(ch, 0, tick);
  else                        pwmHigh.setPWM(ch, 0, tick);
}

// 비차단 출력: 송신 버퍼에 자리가 없으면 버린다(core.statSayDropped로 셈 → PONG saydrop=).
static void sayLine(const char* s) {
  const size_t n = strlen(s) + 2;  // + "\r\n"
  if ((size_t)Serial.availableForWrite() < n) {
    core.statSayDropped++;
    return;
  }
  Serial.println(s);
}

// 배선 점검: 주소에 ACK가 오는지 (setPWM은 실패를 알려주지 않으므로 따로 확인한다)
static bool i2cPresent(uint8_t addr) {
  I2C_BUS.beginTransmission(addr);
  return I2C_BUS.endTransmission() == 0;
}

// PING을 받을 때마다 PCA9685 두 보드의 응답을 실제로 확인해 한 줄 보고한다 (PONG 바로 앞)
static void onFrame(uint8_t type, uint8_t, const uint8_t*, uint8_t) {
  if (type != tv::TYPE_PING) return;
  char line[80];
  snprintf(line, sizeof line, "I2C sda=GP%u scl=GP%u 0x40=%s 0x41=%s map=%s", (unsigned)PIN_SDA, (unsigned)PIN_SCL,
           i2cPresent(ADDR_LOW) ? "ok" : "MISSING", i2cPresent(ADDR_HIGH) ? "ok" : "MISSING",
           tv::CHANNEL_MAP_NAME);
  sayLine(line);
}

void setup() {
  Serial.begin(115200);  // USB CDC라 속도 값은 무시된다(실효 USB FS)

  I2C_BUS.setSDA(PIN_SDA);
  I2C_BUS.setSCL(PIN_SCL);
  I2C_BUS.setClock(I2C_HZ);
  I2C_BUS.begin();
  pwmLow.begin();
  pwmLow.setPWMFreq(PWM_FREQ);
  pwmHigh.begin();
  pwmHigh.setPWMFreq(PWM_FREQ);
  I2C_BUS.setClock(I2C_HZ);  // 라이브러리 begin()이 클럭을 바꾸는 버전에 대비

  core.begin(writePwm, sayLine, millis(), onFrame);
  // 기동 시 전체 홈. 조립 상태를 모르므로 채널마다 25 ms 시간차 — 24개 동시 기동 돌입전류 회피
  for (uint8_t ch = 0; ch < tv::CELL_COUNT; ch++) {
    core.writeChannel(ch, core.currentTick[ch]);
    delay(25);
  }
  core.restartTimers(millis());

  char banner[96];
  snprintf(banner, sizeof banner, "PICO_READY v%u cells=%u proto=bin-crc8 tick=%u jit=%u,%u,%u",
           (unsigned)tv::FW_VERSION, (unsigned)tv::CELL_COUNT, (unsigned)core.tickPeriodMs,
           (unsigned)core.jitterTicks[1], (unsigned)core.jitterTicks[2], (unsigned)core.jitterTicks[3]);
  sayLine(banner);
}

void loop() {
  const uint32_t now = millis();
  int budget = RX_BUDGET;
  while (Serial.available() > 0 && budget-- > 0) core.feed((uint8_t)Serial.read(), now);
  core.poll(millis());
}
