/**
 * 시각-촉각 감각대체 장치 — Raspberry Pi Pico 펌웨어
 *
 * Jetson --USB CDC--> Pico --I2C 400kHz--> PCA9685 x2 --> SG90 x24
 *
 * 하드웨어 실측 기반 상수 (STL 측정에서 유도, src/tactile_vision/geometry.py와 동일)
 *   피니언 12톱니, 모듈 1.3333, 피치원 16.00mm
 *   0.139626 mm/도, 1회전 50.265mm
 *   PWM 50Hz / 4096tick → 1tick = 4.8828us
 *   SG90 500~2400us = tick 102~492 (180도)
 *
 * 기존 펌웨어에서 고친 것
 * ─────────────────────
 *  1. ASCII "GRID:..." + String + strtok  →  고정길이 바이너리 + CRC8 + 상태기계
 *     : 체크섬이 없어 값 하나가 깨지면 서보가 튀었다. String은 장시간 구동에서
 *       RP2040 힙을 단편화한다. readStringUntil은 최대 100ms 블로킹이었다.
 *  2. 전 채널 공통 SAFE_MIN/MAX  →  채널별 home_tick/max_tick 테이블
 *     : 3D 프린트 공차로 24개 유닛의 핀 높이가 다르다. 기존 250~490tick은
 *       실측 68.3~179.3도(15.5mm)로 가용 20.9mm 중 5.4mm를 버렸다.
 *  3. 슬루 제한 없음  →  틱/주기 상한
 *     : 목표값으로 즉시 내던지면 24개가 동시에 최대토크를 끌어 브라운아웃이 난다.
 *  4. 동시 구동 제한 없음  →  max_simultaneous
 *     : SG90 이동전류 ~250mA x 24 = 6A. 3S→5V 벅 정격을 넘긴다.
 *  5. 매 업데이트 전 채널 setPWM  →  변경된 채널만 기록
 *     : I2C 400kHz에서 24채널 전체 쓰기는 약 2.7ms로 주기의 절반을 먹는다.
 *
 * 배선: GP4=SDA, GP5=SCL / PCA9685 0x40(ch0~11), 0x41(ch12~23)
 *       외부 5V → PCA9685 V+ (서보 전원, Pico 3V3와 분리)
 *
 * 의존: Adafruit PWM Servo Driver Library
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ─────────────────────────────────────────────────────────
// 컴파일 시 상수
// ─────────────────────────────────────────────────────────
static constexpr uint8_t  CELL_COUNT   = 24;
static constexpr uint8_t  BOARD_CH     = 12;
static constexpr uint8_t  ADDR_LOW     = 0x40;
static constexpr uint8_t  ADDR_HIGH    = 0x41;
static constexpr uint32_t I2C_HZ       = 400000;
static constexpr uint16_t PWM_FREQ     = 50;

// SG90 절대 안전 한계. 캘리브레이션 값도 이 범위로 클램프된다.
// 400us/2600us를 넘으면 기구 간섭이나 스톨로 기어가 깨진다.
static constexpr uint16_t TICK_HARD_MIN = 82;   //  400us
static constexpr uint16_t TICK_HARD_MAX = 533;  // 2600us

// 서보 업데이트 주기
static constexpr uint32_t UPDATE_PERIOD_MS = 5;   // 200Hz

// 슬루 제한: SG90 부하시 약 360도/s = 778 tick/s → 5ms당 약 4tick.
// 이보다 크게 명령해도 서보는 못 따라오고 전류만 증가한다.
static constexpr uint16_t SLEW_TICKS_PER_UPDATE = 4;

// 통신 단절 failsafe
static constexpr uint32_t COMM_TIMEOUT_MS = 500;

// jitter(위험 알림) 기본 주기. 반주기마다 부호가 바뀐다 → 약 6.7Hz.
static constexpr uint16_t JITTER_HALF_PERIOD_MS_DEFAULT = 75;

// jitter 진폭: 위험도 1/2/3 → 1.0/1.5/2.0mm
// 1mm = 7.162도 = 7.162 * (389tick/180도) = 15.5tick
static constexpr uint16_t JITTER_TICKS[4] = {0, 15, 23, 31};

// ─────────────────────────────────────────────────────────
// 프로토콜 (src/tactile_vision/protocol.py와 반드시 일치)
// ─────────────────────────────────────────────────────────
static constexpr uint8_t SYNC0 = 0xAA;
static constexpr uint8_t SYNC1 = 0x55;

static constexpr uint8_t TYPE_GRID       = 0x01;
static constexpr uint8_t TYPE_RAW_TICK   = 0x02;
static constexpr uint8_t TYPE_HOME_ALL   = 0x03;
static constexpr uint8_t TYPE_SET_LIMIT  = 0x04;
static constexpr uint8_t TYPE_PING       = 0x05;
static constexpr uint8_t TYPE_SET_CONFIG = 0x06;

static constexpr uint8_t FLAG_ESTOP = 0x01;

static constexpr uint8_t MAX_PAYLOAD = 64;

// ─────────────────────────────────────────────────────────
// 상태
// ─────────────────────────────────────────────────────────
Adafruit_PWMServoDriver pwmLow (ADDR_LOW);
Adafruit_PWMServoDriver pwmHigh(ADDR_HIGH);

struct ChannelCal {
  uint16_t homeTick;
  uint16_t maxTick;
};

// 기하 계산 기본값 (10도 여유 + 150도 스트로크 → 605us~2188us)
// 실제 값은 Jetson이 기동 시 SET_LIMIT으로 채널마다 덮어쓴다.
ChannelCal cal[CELL_COUNT];

uint16_t targetTick[CELL_COUNT];   // 목표 위치 (jitter 제외)
uint16_t currentTick[CELL_COUNT];  // 현재 명령된 위치
uint16_t writtenTick[CELL_COUNT];  // 실제 PCA9685에 쓴 값 (중복 쓰기 방지)
uint8_t  cellLevel[CELL_COUNT];
uint8_t  cellHazard[CELL_COUNT];

uint8_t  runtimeLevels      = 8;
uint8_t  maxSimultaneous    = 8;
uint16_t jitterHalfPeriodMs = JITTER_HALF_PERIOD_MS_DEFAULT;

bool     estop        = false;
uint32_t lastRxMs     = 0;
uint32_t lastUpdateMs = 0;
uint32_t lastStatMs   = 0;

// 통계
uint32_t statFrames  = 0;
uint32_t statCrcErr  = 0;
uint32_t statResync  = 0;
uint32_t statDropped = 0;
uint8_t  lastSeq     = 0;
bool     seqInit     = false;

// ─────────────────────────────────────────────────────────
// CRC8 (poly 0x07, init 0x00) — Python 구현과 동일
// ─────────────────────────────────────────────────────────
static uint8_t crc8(const uint8_t* data, size_t len) {
  uint8_t crc = 0x00;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

// ─────────────────────────────────────────────────────────
// 유틸
// ─────────────────────────────────────────────────────────
static inline uint16_t clampTick(uint16_t t) {
  if (t < TICK_HARD_MIN) return TICK_HARD_MIN;
  if (t > TICK_HARD_MAX) return TICK_HARD_MAX;
  return t;
}

static inline uint16_t tickForLevel(uint8_t ch, uint8_t level) {
  const ChannelCal& c = cal[ch];
  uint8_t top = (runtimeLevels > 1) ? (runtimeLevels - 1) : 1;
  if (level > top) level = top;
  // 정수 연산으로 선형 보간. home > max(역방향 장착)도 자연히 처리된다.
  int32_t span = (int32_t)c.maxTick - (int32_t)c.homeTick;
  int32_t t = (int32_t)c.homeTick + (span * (int32_t)level) / (int32_t)top;
  if (t < 0) t = 0;
  if (t > 4095) t = 4095;
  return clampTick((uint16_t)t);
}

static inline void writeChannel(uint8_t ch, uint16_t tick) {
  if (writtenTick[ch] == tick) return;  // I2C 트래픽 절감
  writtenTick[ch] = tick;
  if (ch < BOARD_CH) pwmLow.setPWM(ch, 0, tick);
  else               pwmHigh.setPWM(ch - BOARD_CH, 0, tick);
}

static void homeAll(bool immediate) {
  for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
    cellLevel[ch]  = 0;
    cellHazard[ch] = 0;
    targetTick[ch] = clampTick(cal[ch].homeTick);
    if (immediate) {
      currentTick[ch] = targetTick[ch];
      writeChannel(ch, currentTick[ch]);
    }
  }
}

// ─────────────────────────────────────────────────────────
// 서보 갱신: 슬루 제한 + 동시구동 제한 + jitter
// ─────────────────────────────────────────────────────────
static void updateServos() {
  // jitter 부호: 반주기마다 토글
  const bool jitterPhase = ((millis() / jitterHalfPeriodMs) & 1) != 0;

  // 이번 주기에 실제로 움직여야 하는 채널을 모은다.
  // 우선순위: (1) 상승(장애물 접근 = 위험 신호), (2) 오차 큰 순.
  uint8_t  cand[CELL_COUNT];
  int32_t  err[CELL_COUNT];
  bool     rising[CELL_COUNT];
  uint8_t  nCand = 0;

  for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
    uint16_t want = targetTick[ch];

    // 위험 셀은 목표 주변으로 진동시킨다. 진동은 슬루/동시구동 제한을 거치므로
    // 추가 전류는 이미 상한 안에 들어온다.
    if (!estop && cellHazard[ch] > 0) {
      uint8_t hz = cellHazard[ch] > 3 ? 3 : cellHazard[ch];
      int32_t amp = JITTER_TICKS[hz];
      // 홈 방향으로만 흔든다. 최상단에서 위로 더 밀면 기구에 부딪힐 수 있다.
      int32_t sign = (cal[ch].maxTick >= cal[ch].homeTick) ? -1 : +1;
      if (jitterPhase) want = clampTick((uint16_t)((int32_t)want + sign * amp));
    }

    int32_t d = (int32_t)want - (int32_t)currentTick[ch];
    if (d == 0) continue;
    // 상승 = 핀이 올라가는 방향. 역방향 장착을 고려해 부호를 판별한다.
    bool up = (cal[ch].maxTick >= cal[ch].homeTick) ? (d > 0) : (d < 0);
    cand[nCand]   = ch;
    err[nCand]    = d < 0 ? -d : d;
    rising[nCand] = up;
    nCand++;
  }

  // 부분 선택 정렬 — 최대 maxSimultaneous개만 뽑으므로 O(n*k)면 충분하다.
  uint8_t budget = maxSimultaneous;
  if (budget == 0) budget = 1;
  if (budget > nCand) budget = nCand;

  bool used[CELL_COUNT];
  for (uint8_t i = 0; i < nCand; i++) used[i] = false;

  for (uint8_t pick = 0; pick < budget; pick++) {
    int8_t best = -1;
    for (uint8_t i = 0; i < nCand; i++) {
      if (used[i]) continue;
      if (best < 0) { best = i; continue; }
      // 상승 우선, 같으면 오차 큰 쪽
      if (rising[i] != rising[best]) { if (rising[i]) best = i; }
      else if (err[i] > err[best])   { best = i; }
    }
    if (best < 0) break;
    used[best] = true;

    uint8_t ch = cand[best];
    int32_t d = (int32_t)targetTick[ch] - (int32_t)currentTick[ch];
    // jitter 적용된 목표를 다시 계산하지 않고 원목표로 추종해도 무방하다.
    // (jitter는 다음 주기에 다시 반영된다)
    if (!estop && cellHazard[ch] > 0 && jitterPhase) {
      uint8_t hz = cellHazard[ch] > 3 ? 3 : cellHazard[ch];
      int32_t sign = (cal[ch].maxTick >= cal[ch].homeTick) ? -1 : +1;
      int32_t want = (int32_t)targetTick[ch] + sign * (int32_t)JITTER_TICKS[hz];
      d = want - (int32_t)currentTick[ch];
    }
    int32_t step = d;
    if (step >  (int32_t)SLEW_TICKS_PER_UPDATE) step =  (int32_t)SLEW_TICKS_PER_UPDATE;
    if (step < -(int32_t)SLEW_TICKS_PER_UPDATE) step = -(int32_t)SLEW_TICKS_PER_UPDATE;
    currentTick[ch] = clampTick((uint16_t)((int32_t)currentTick[ch] + step));
    writeChannel(ch, currentTick[ch]);
  }
}

// ─────────────────────────────────────────────────────────
// 프레임 처리
// ─────────────────────────────────────────────────────────
static void handleGrid(const uint8_t* p, uint8_t len) {
  if (len != CELL_COUNT + 1) {
    Serial.print(F("ERR grid_len=")); Serial.println(len);
    return;
  }
  uint8_t flags = p[CELL_COUNT];
  if (flags & FLAG_ESTOP) {
    estop = true;
    homeAll(true);
    Serial.println(F("ESTOP"));
    return;
  }
  estop = false;
  for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
    cellLevel[ch]  = p[ch] & 0x0F;
    cellHazard[ch] = (p[ch] >> 4) & 0x0F;
    targetTick[ch] = tickForLevel(ch, cellLevel[ch]);
  }
}

static void handleFrame(uint8_t type, uint8_t seq, const uint8_t* p, uint8_t len) {
  statFrames++;
  lastRxMs = millis();

  // 시퀀스 유실 감지 (패킷 손실률 측정용)
  if (seqInit) {
    uint8_t expect = (uint8_t)(lastSeq + 1);
    if (seq != expect) statDropped += (uint8_t)(seq - expect);
  }
  lastSeq = seq;
  seqInit = true;

  switch (type) {
    case TYPE_GRID:
      handleGrid(p, len);
      break;

    case TYPE_RAW_TICK: {
      if (len != 3) { Serial.println(F("ERR raw_len")); break; }
      uint8_t  ch   = p[0];
      uint16_t tick = ((uint16_t)p[1] << 8) | p[2];
      if (ch >= CELL_COUNT) { Serial.println(F("ERR raw_ch")); break; }
      // 캘리브레이션 모드: 슬루 제한을 우회해 즉시 인가한다.
      uint16_t t = clampTick(tick);
      targetTick[ch] = t;
      currentTick[ch] = t;
      writeChannel(ch, t);
      Serial.print(F("TICK ")); Serial.print(ch);
      Serial.print(' ');        Serial.println(t);
      break;
    }

    case TYPE_HOME_ALL:
      estop = false;
      homeAll(true);
      Serial.println(F("HOMED"));
      break;

    case TYPE_SET_LIMIT: {
      if (len != 5) { Serial.println(F("ERR lim_len")); break; }
      uint8_t  ch = p[0];
      uint16_t hm = ((uint16_t)p[1] << 8) | p[2];
      uint16_t mx = ((uint16_t)p[3] << 8) | p[4];
      if (ch >= CELL_COUNT) { Serial.println(F("ERR lim_ch")); break; }
      cal[ch].homeTick = clampTick(hm);
      cal[ch].maxTick  = clampTick(mx);
      targetTick[ch]   = tickForLevel(ch, cellLevel[ch]);
      Serial.print(F("LIMIT ")); Serial.print(ch);
      Serial.print(' '); Serial.print(cal[ch].homeTick);
      Serial.print(' '); Serial.println(cal[ch].maxTick);
      break;
    }

    case TYPE_PING:
      Serial.print(F("PONG f=")); Serial.print(statFrames);
      Serial.print(F(" crc=")); Serial.print(statCrcErr);
      Serial.print(F(" resync=")); Serial.print(statResync);
      Serial.print(F(" drop=")); Serial.println(statDropped);
      break;

    case TYPE_SET_CONFIG: {
      if (len != 4) { Serial.println(F("ERR cfg_len")); break; }
      uint8_t lv = p[0];
      uint8_t ms = p[1];
      uint16_t jp = ((uint16_t)p[2] << 8) | p[3];
      if (lv >= 2 && lv <= 16) runtimeLevels = lv;
      if (ms >= 1 && ms <= CELL_COUNT) maxSimultaneous = ms;
      if (jp >= 20 && jp <= 1000) jitterHalfPeriodMs = jp;
      for (uint8_t ch = 0; ch < CELL_COUNT; ch++)
        targetTick[ch] = tickForLevel(ch, cellLevel[ch]);
      Serial.print(F("CFG lv=")); Serial.print(runtimeLevels);
      Serial.print(F(" sim=")); Serial.print(maxSimultaneous);
      Serial.print(F(" jit=")); Serial.println(jitterHalfPeriodMs);
      break;
    }

    default:
      Serial.print(F("ERR type=")); Serial.println(type);
  }
}

// ─────────────────────────────────────────────────────────
// 수신 상태기계 — 블로킹 없음, 동적할당 없음
// ─────────────────────────────────────────────────────────
enum RxState : uint8_t { WAIT_S0, WAIT_S1, GET_TYPE, GET_SEQ, GET_LEN, GET_PAYLOAD, GET_CRC };

RxState rxState = WAIT_S0;
uint8_t rxType, rxSeq, rxLen, rxIdx;
uint8_t rxBuf[MAX_PAYLOAD];

static void pumpSerial() {
  // 한 번에 처리할 바이트 수를 제한해 서보 갱신 주기를 지키게 한다.
  int budget = 512;
  while (Serial.available() > 0 && budget-- > 0) {
    uint8_t b = (uint8_t)Serial.read();
    switch (rxState) {
      case WAIT_S0:
        if (b == SYNC0) rxState = WAIT_S1;
        else statResync++;
        break;
      case WAIT_S1:
        if (b == SYNC1)      rxState = GET_TYPE;
        else if (b == SYNC0) rxState = WAIT_S1;  // 0xAA 0xAA 0x55 대응
        else { statResync++; rxState = WAIT_S0; }
        break;
      case GET_TYPE: rxType = b; rxState = GET_SEQ; break;
      case GET_SEQ:  rxSeq  = b; rxState = GET_LEN; break;
      case GET_LEN:
        rxLen = b;
        if (rxLen > MAX_PAYLOAD) { statCrcErr++; rxState = WAIT_S0; break; }
        rxIdx = 0;
        rxState = (rxLen == 0) ? GET_CRC : GET_PAYLOAD;
        break;
      case GET_PAYLOAD:
        rxBuf[rxIdx++] = b;
        if (rxIdx >= rxLen) rxState = GET_CRC;
        break;
      case GET_CRC: {
        uint8_t body[3 + MAX_PAYLOAD];
        body[0] = rxType; body[1] = rxSeq; body[2] = rxLen;
        for (uint8_t i = 0; i < rxLen; i++) body[3 + i] = rxBuf[i];
        if (crc8(body, 3 + rxLen) == b) {
          handleFrame(rxType, rxSeq, rxBuf, rxLen);
        } else {
          statCrcErr++;
        }
        rxState = WAIT_S0;
        break;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  Wire.setSDA(4);
  Wire.setSCL(5);
  Wire.begin();
  Wire.setClock(I2C_HZ);

  pwmLow.begin();
  pwmLow.setPWMFreq(PWM_FREQ);
  pwmHigh.begin();
  pwmHigh.setPWMFreq(PWM_FREQ);

  // 기하 계산 기본값. Jetson이 SET_LIMIT으로 채널별 실측값을 덮어쓴다.
  for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
    cal[ch].homeTick = 124;  //  605us (10도)
    cal[ch].maxTick  = 448;  // 2188us (160도) → 스트로크 20.9mm
    writtenTick[ch]  = 0xFFFF;  // 첫 쓰기를 강제
  }

  // 기동 시 전체 홈. 조립 상태를 모르므로 한 채널씩 시간차를 두어
  // 24개 동시 기동 돌입전류를 피한다.
  for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
    cellLevel[ch]   = 0;
    cellHazard[ch]  = 0;
    targetTick[ch]  = cal[ch].homeTick;
    currentTick[ch] = cal[ch].homeTick;
    writeChannel(ch, cal[ch].homeTick);
    delay(25);
  }

  lastRxMs = millis();
  lastUpdateMs = lastRxMs;
  Serial.println(F("PICO_READY v2 cells=24 proto=bin-crc8"));
}

void loop() {
  pumpSerial();

  uint32_t now = millis();

  // failsafe: 통신 단절 시 홈 복귀. 슬루 제한을 통해 천천히 내린다.
  if (now - lastRxMs > COMM_TIMEOUT_MS) {
    for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
      cellLevel[ch]  = 0;
      cellHazard[ch] = 0;
      targetTick[ch] = clampTick(cal[ch].homeTick);
    }
    // 재진입 방지를 위해 타임아웃을 갱신하되, 상태 메시지는 1초에 한 번만
    if (now - lastStatMs > 1000) {
      Serial.println(F("FAILSAFE no_comm"));
      lastStatMs = now;
    }
  }

  if (now - lastUpdateMs >= UPDATE_PERIOD_MS) {
    lastUpdateMs = now;
    updateServos();
  }
}
