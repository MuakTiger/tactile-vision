/**
 * 촉각 장치 서보 캘리브레이션 — Arduino Uno 스윕 스케치
 *
 * 서보 1개를 D7에 직접 연결해 핀의 "최저점(멀리 있는 출력) → 최대 → 다시 최저 → 쉼"을 반복한다.
 * 속도 측정(스마트폰 슬로모션)과 채널별 home/max tick 찾기용. 값은 PCA9685와 같은 tick 단위다.
 *
 *   1 tick = 4.8828125 µs (50 Hz, 4096분할)        SG90 명목: 500 µs = 0도, 2400 µs = 180도
 *   기본: 시작 232 tick ≈ 60도(1133 µs), 범위 140 tick(684 µs ≈ 64.8도 ≈ 9.0 mm) → 끝 372 tick ≈ 124.7도
 *   핀 이동량: 피니언 피치원 16.00 mm 기준 1도 = 0.1396 mm
 *
 * 배선
 *   서보 신호(주황/노랑) → D7
 *   서보 전원(빨강)      → 외부 5 V (스텝다운)   ※ Uno 5V 핀은 서보가 걸리면 전류가 모자랄 수 있다
 *   서보 GND(갈색)       → 외부 전원 GND와 Uno GND를 반드시 함께 연결 (공통 GND)
 *
 * 동작 (기본)
 *   시작 → 최저점으로 이동, 1 s 대기 → 반복 { 최대로 이동 → 유지 → 최저로 이동 → 유지 → 쉼 1 s }
 *   D13 LED: 최대로 보내는 순간 켜지고, 최저로 보내는 순간 꺼진다 — 슬로모션 영상에서 명령 시각 기준으로 쓴다.
 *   기본 이동 방식은 "점프": 목표 펄스를 한 번에 보내고 서보가 자기 속도로 간다(실제 속도 측정용).
 *
 * 시리얼 (115200 baud, 줄 끝 아무거나) — 바뀐 값은 다음 동작부터 적용
 *   ?        도움말과 현재 설정
 *   s N      시작(최저점) tick            w N   범위 tick          h N   이동 후 유지 ms
 *   p N      한 주기 끝 쉬는 시간 ms      m 0   점프(서보 자기 속도)  m 1   슬루(펌웨어와 같은 5 ms당 4 tick)
 *   t N      N tick으로 가서 멈춤(조그)    +  -  조그 ±5 tick        r     자동 스윕 다시 시작
 *   x        스윕 멈춤(현재 위치 유지)
 *
 * 안전: 모든 명령은 펌웨어와 같은 절대 한계 82~533 tick(400~2600 µs)으로 제한된다.
 * 빌드/업로드: tools/upload_uno_sweep.sh   (FQBN arduino:avr:uno, Servo 라이브러리는 코어에 포함)
 */

#include <Servo.h>

static const uint8_t  SERVO_PIN     = 7;
static const uint16_t HARD_MIN_TICK = 82;    //  400 µs — 펌웨어 TICK_HARD_MIN과 같음
static const uint16_t HARD_MAX_TICK = 533;   // 2600 µs — 펌웨어 TICK_HARD_MAX와 같음
static const uint16_t SLEW_TICKS    = 4;     // 슬루 모드: 5 ms마다 4 tick (= 펌웨어 SLEW_TICKS_PER_UPDATE)
static const uint16_t SLEW_MS       = 5;
static const uint16_t INIT_HOLD_MS  = 1000;  // 시작 직후 최저점에서 기다리는 시간
static const uint16_t JOG_STEP      = 5;

// 설정 (시리얼로 바꿀 수 있음)
uint16_t startTick = 232;   // 60도: 500 + 60/180*1900 = 1133 µs = 232 tick
uint16_t spanTicks = 140;   // 2026-10-04 결정: 0.2 s 끝→끝 기준으로 140 tick
uint16_t holdMs    = 500;   // 이동 명령 뒤 유지 시간. 0.2 s 끝→끝 시험은 h 200
uint16_t pauseMs   = 1000;  // 한 주기 끝 쉼
bool     slewMode  = false;

Servo servo;
uint16_t curTick;           // 지금 명령 중인 위치
uint16_t tgtTick;           // 목표
uint32_t lastSlewMs = 0;

enum Phase : uint8_t { P_INIT, P_UP, P_DOWN, P_REST, P_STOP };
Phase    phase = P_INIT;
bool     armed = false;     // 목표에 도착해 유지 시간 재는 중
uint32_t armedAt = 0;
uint32_t cycle = 0;

char   lineBuf[24];
uint8_t lineLen = 0;

static uint16_t clampTick(long t) {
  if (t < HARD_MIN_TICK) return HARD_MIN_TICK;
  if (t > HARD_MAX_TICK) return HARD_MAX_TICK;
  return (uint16_t)t;
}
static uint16_t tickToUs(uint16_t t) { return (uint16_t)(((uint32_t)t * 625UL + 64UL) / 128UL); }
static float    tickToDeg(uint16_t t) { return ((float)t * 4.8828125f - 500.0f) / 1900.0f * 180.0f; }
static uint16_t highTick() { return clampTick((long)startTick + spanTicks); }
static uint16_t lowTick()  { return clampTick(startTick); }

static void writeTick(uint16_t t) { servo.writeMicroseconds(tickToUs(t)); }

static void setTarget(uint16_t t) {
  tgtTick = clampTick(t);
  armed = false;
  if (!slewMode) {           // 점프: 한 번에 보낸다
    curTick = tgtTick;
    writeTick(curTick);
  }
}

static void printTick(uint16_t t) {
  Serial.print(t); Serial.print(F(" tick = ")); Serial.print(tickToUs(t));
  Serial.print(F(" us = ")); Serial.print(tickToDeg(t), 1); Serial.print(F(" deg"));
}

static void printSettings() {
  Serial.print(F("start ")); printTick(lowTick()); Serial.println();
  Serial.print(F("max   ")); printTick(highTick()); Serial.println();
  const uint16_t span = highTick() - lowTick();
  Serial.print(F("span  ")); Serial.print(span); Serial.print(F(" tick = "));
  Serial.print((float)span * 4.8828125f, 0); Serial.print(F(" us = "));
  Serial.print((float)span / 2.1619f, 1); Serial.print(F(" deg = "));
  Serial.print((float)span / 2.1619f * 0.139626f, 1); Serial.println(F(" mm (16 mm pinion)"));
  Serial.print(F("hold ")); Serial.print(holdMs); Serial.print(F(" ms, pause ")); Serial.print(pauseMs);
  Serial.print(F(" ms, mode ")); Serial.println(slewMode ? F("slew 4 tick/5 ms") : F("jump (servo's own speed)"));
}

static void printHelp() {
  Serial.println(F("--- uno_servo_sweep (D7) ---"));
  printSettings();
  Serial.println(F("s N start tick | w N span tick | h N hold ms | p N pause ms | m 0 jump / m 1 slew"));
  Serial.println(F("t N go to tick and stop | + - jog 5 tick | r resume sweep | x stop | ? help"));
}

static void enter(Phase p) {
  phase = p;
  armed = false;
  switch (p) {
    case P_INIT:
      setTarget(lowTick());
      Serial.print(F("INIT  -> low ")); printTick(tgtTick); Serial.println();
      break;
    case P_UP:
      digitalWrite(LED_BUILTIN, HIGH);        // 명령 시각 표시: 최대로 보냄
      setTarget(highTick());                  // 서보 명령을 LED 바로 뒤에, 출력은 그 다음
      Serial.print(F("UP    -> max ")); printTick(tgtTick); Serial.println();
      break;
    case P_DOWN:
      digitalWrite(LED_BUILTIN, LOW);         // 명령 시각 표시: 최저로 보냄
      setTarget(lowTick());
      Serial.print(F("DOWN  -> low ")); printTick(tgtTick); Serial.println();
      break;
    case P_REST:
      cycle++;
      Serial.print(F("REST  cycle ")); Serial.print(cycle); Serial.print(F(", ")); Serial.print(pauseMs); Serial.println(F(" ms"));
      armed = true; armedAt = millis();       // 쉼은 도착 여부와 무관하게 바로 센다
      break;
    case P_STOP:
      Serial.print(F("STOP  at ")); printTick(curTick); Serial.println();
      break;
  }
}

static void jogTo(long t) {
  const bool wasSlew = slewMode;
  slewMode = false;                           // 조그는 항상 즉시
  setTarget(clampTick(t));
  slewMode = wasSlew;
  phase = P_STOP;
  Serial.print(F("JOG   ")); printTick(curTick); Serial.println();
}

static void handleLine(char* s) {
  while (*s == ' ') s++;
  const char c = *s;
  if (c == 0) return;
  long n = atol(s + 1);
  switch (c) {
    case '?': printHelp(); break;
    case 's': startTick = clampTick(n); printSettings(); break;
    case 'w': if (n >= 2 && n <= 450) spanTicks = (uint16_t)n; printSettings(); break;
    case 'h': if (n >= 20 && n <= 5000) holdMs = (uint16_t)n; printSettings(); break;
    case 'p': if (n >= 0 && n <= 20000) pauseMs = (uint16_t)n; printSettings(); break;
    case 'm': slewMode = (n != 0); printSettings(); break;
    case 't': jogTo(n); break;
    case '+': jogTo((long)curTick + JOG_STEP); break;
    case '-': jogTo((long)curTick - JOG_STEP); break;
    case 'r': cycle = 0; enter(P_INIT); break;
    case 'x': enter(P_STOP); break;
    default: Serial.print(F("? unknown: ")); Serial.println(s); break;
  }
}

static void pollSerial() {
  while (Serial.available() > 0) {
    const char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (lineLen > 0) { lineBuf[lineLen] = 0; handleLine(lineBuf); lineLen = 0; }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = ch;
    } else {
      lineLen = 0;                            // 너무 긴 줄은 버린다
    }
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);
  servo.attach(SERVO_PIN, 400, 2600);         // 펄스 한계를 펌웨어 하드 한계와 같게 (라이브러리 기본은 544~2400)
  curTick = tgtTick = lowTick();
  printHelp();
  enter(P_INIT);
}

void loop() {
  pollSerial();
  const uint32_t now = millis();

  // 슬루 모드: 5 ms마다 4 tick씩 목표로
  if (curTick != tgtTick && slewMode && (now - lastSlewMs) >= SLEW_MS) {
    lastSlewMs = now;
    const int32_t d = (int32_t)tgtTick - (int32_t)curTick;
    curTick = (uint16_t)((int32_t)curTick + (d > (int32_t)SLEW_TICKS ? SLEW_TICKS : (d < -(int32_t)SLEW_TICKS ? -(int32_t)SLEW_TICKS : d)));
    writeTick(curTick);
  }

  if (phase == P_STOP) return;

  // 목표에 도착하면 유지 시간을 센다 (점프 모드는 명령 즉시 "도착"으로 본다)
  if (!armed && curTick == tgtTick) { armed = true; armedAt = now; }
  if (!armed) return;

  const uint32_t waitMs = (phase == P_INIT) ? INIT_HOLD_MS : (phase == P_REST ? pauseMs : holdMs);
  if ((now - armedAt) < waitMs) return;

  switch (phase) {
    case P_INIT: enter(P_UP);   break;
    case P_UP:   enter(P_DOWN); break;
    case P_DOWN: enter(P_REST); break;
    case P_REST: enter(P_UP);   break;
    default: break;
  }
}
