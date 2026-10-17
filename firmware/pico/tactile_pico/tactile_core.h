/**
 * 촉각 장치 Pico 펌웨어 핵심 로직 v3 — 하드웨어 독립 (C++17, 표준 C 헤더만 사용).
 *
 * 실기(tactile_pico.ino)와 PC 시험(../test_harness.cpp)이 이 파일 하나를 같이 쓴다.
 * 하드웨어 접근은 콜백 두 개뿐이다: PWM 쓰기(WriteFn), 상태 한 줄 출력(SayFn).
 *
 * v3 동작 (2026-10-03)
 * ───────────────────
 *  1. 박자 적용 — 젯슨은 계산이 끝날 때마다 GRID를 보낸다(약 15 Hz). Pico는 받은 값을
 *     '최신 값'으로 보관만 하고, 자기 박자(tickPeriodMs, 기본 200 ms = 초당 5회)마다
 *     그 순간의 최신 값 하나를 핀 목표로 적용한다. 적용 전에 덮어써진 값은 statSuperseded.
 *  2. 위험도 진동 — 박자를 4등분해 2·4번째 1/4 구간(200 ms 박자면 50~100, 150~200 ms)에만
 *     위험 셀을 jitterTicks[위험도]만큼 비켜 놓는다 → 거리 갱신 한 번 사이에 진동 2번
 *     = 거리 갱신의 2배 빈도. 기본 2/4/6 tick ≈ 1/2/3도(SG90 명목 1도 = 2.162 tick).
 *     방향은 홈 쪽(핀이 살짝 내려감 — 위로 밀어 기구 끝에 부딪히지 않게). 이미 홈이라
 *     더 내려갈 수 없으면 위로. 채널 보정 범위[home, max] 밖으로는 가지 않는다.
 *  3. 즉시 처리 — ESTOP 플래그, HOME_ALL, RAW_TICK(캘리브레이션), SET_LIMIT, SET_CONFIG는
 *     박자를 기다리지 않는다. ESTOP·HOME_ALL·RAW_TICK은 보관 중인 최신 값도 버린다.
 *     ESTOP은 즉시 홈(점프), HOME_ALL은 천천히 홈(아래 4).
 *  4. 천천히 홈 복귀 (2026-10-05) — 젯슨 프로그램이 끝날 때(HOME_ALL)나 유효 프레임이 COMM_TIMEOUT_MS
 *     (500 ms) 동안 없을 때(failsafe: 프로그램이 죽거나 멈춤) 전체 목표를 홈(설정 범위의 맨 아래)으로 두고
 *     RETURN_STEP_MS(10 ms)마다 1 tick씩 내린다(≈46도/s, 범위 140 tick을 1.4 s). 보관 값 폐기.
 *     다음 핀 값이 박자에 적용되면 보통 속도로 돌아간다. 캘리브레이션 도구는 KEEPALIVE(0x07)로 failsafe를 막는다.
 *     ※ Pico 전원이 끊기면(USB를 뽑는 등) 신호가 사라져 서보가 힘을 잃는다 — 소프트웨어로 내릴 수 없다.
 *       다시 켜지면 칸별 기본 home(homeDefault)으로 간다(서보 위치를 모르므로 이때는 서보 자체 속도).
 *  5. 슬루 제한(5 ms당 4 tick)과 동시 구동 제한(maxSimultaneous, 상승 우선)은 v2와 같다.
 *
 * 프로토콜은 src/tactile_vision/protocol.py와 반드시 일치해야 한다.
 */
#pragma once

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

namespace tv {

// ─────────────────────────────────────────────────────────
// 상수
// ─────────────────────────────────────────────────────────
constexpr uint8_t FW_VERSION = 3;
constexpr uint8_t CELL_COUNT = 24;

// 칸(화면 순서, 행 우선 0~23) → PCA9685 채널 — 2026-10-05 배선.
// 칸 0~11(1~2행)은 0x40 보드, 칸 12~23(3~4행)은 0x41 보드. 보드마다 왼쪽 위부터 세로 2칸이 한 세트:
//   채널 = 열 × 2 + (보드 안에서 위 줄 0 / 아래 줄 1)
//     위 줄  0 2 4 6 8 10        (채널 0~5 = 왼손 열 0~2, 6~11 = 오른손 열 3~5)
//     아래줄 1 3 5 7 9 11
// src/tactile_vision/geometry.py CELL_TO_PCA와 같아야 한다(tools/test_firmware.py가 하네스 --map으로 확인).
constexpr uint8_t CELLS_PER_BOARD = 12;
constexpr uint8_t CELL_TO_CHANNEL[CELL_COUNT] = {
    0, 2, 4, 6, 8, 10,   // 1행 → 0x40
    1, 3, 5, 7, 9, 11,   // 2행 → 0x40
    0, 2, 4, 6, 8, 10,   // 3행 → 0x41
    1, 3, 5, 7, 9, 11,   // 4행 → 0x41
};
constexpr const char* CHANNEL_MAP_NAME = "colpairs";
constexpr uint8_t boardOf(uint8_t cell) { return cell < CELLS_PER_BOARD ? 0 : 1; }

// 보드마다 채널 0~11이 정확히 한 번씩 쓰이는지 컴파일 시 확인 (중복·누락 배선 방지)
constexpr bool channelMapIsPermutation() {
  for (uint8_t b = 0; b < 2; b++) {
    uint16_t seen = 0;
    for (uint8_t i = 0; i < CELLS_PER_BOARD; i++) {
      const uint8_t ch = CELL_TO_CHANNEL[b * CELLS_PER_BOARD + i];
      if (ch >= CELLS_PER_BOARD || (seen & (1u << ch))) return false;
      seen |= (uint16_t)(1u << ch);
    }
  }
  return true;
}
static_assert(channelMapIsPermutation(), "CELL_TO_CHANNEL: 보드마다 채널 0~11이 한 번씩이어야 한다");

// SG90 절대 안전 한계. 캘리브레이션 값도 이 범위로 클램프된다 (400~2600 us).
constexpr uint16_t TICK_HARD_MIN = 82;
constexpr uint16_t TICK_HARD_MAX = 533;
// 채널 기본값 — 이 장치(2026-10-05): 서보를 역방향으로 조립(tick이 커질수록 핀이 내려감), 사용자 범위 232~372 tick.
// 젯슨이 연결되면 calib/servo_calib.json 값을 SET_LIMIT으로 덮어쓴다. 젯슨 연결 전에 Pico가 켜지면 핀이 칸별 기본 home으로 간다
// → 캘리브레이션을 바꾸면 아래 값도 함께 바꿀 것(tools/test_firmware.py가 calib 파일·calibration.py와 같은지 확인).
constexpr uint16_t HOME_TICK_DEFAULT = 372;  // 1816 us (124.7도) — 공통 기준(영점 보정 전) 핀 맨 아래 (레벨 0, 먼 곳)
constexpr uint16_t MAX_TICK_DEFAULT = 232;   // 1133 us (60.0도)  — 공통 기준(영점 보정 전) 핀 맨 위 (가까운 곳)
// 칸별 영점 보정 (2026-10-17, tools/pin_zero_gui.py): 0단계가 패드 표면과 같아지게 home·max에 같이 더하는 tick.
// + = tick 증가(이 장치는 핀이 내려감). 칸 순서 = 화면 위 줄 왼쪽부터. PC 하네스 기본 빌드(-DTV_UNIFORM_DEFAULTS)는
// 모든 칸 0으로 두고 동작 규칙만 시험한다.
#ifdef TV_UNIFORM_DEFAULTS
constexpr int16_t ZERO_SHIFT_DEFAULT[CELL_COUNT] = {};
#else
constexpr int16_t ZERO_SHIFT_DEFAULT[CELL_COUNT] = {
     50,  10,   0, -30,  25,  80,
      0,  -5, -15, -55,   0,  20,
    -50, -10, -15, -40, -50,   0,
      0,  20,  -5, -25,  45,  50,
};
#endif
constexpr uint16_t homeDefault(uint8_t cell) { return (uint16_t)(HOME_TICK_DEFAULT + ZERO_SHIFT_DEFAULT[cell]); }
constexpr uint16_t maxDefault(uint8_t cell) { return (uint16_t)(MAX_TICK_DEFAULT + ZERO_SHIFT_DEFAULT[cell]); }

// 천천히 홈 복귀 — 종료(HOME_ALL)·통신 끊김(failsafe) 때. RETURN_STEP_MS마다 1 tick = 100 tick/s ≈ 46도/s
// → 범위 140 tick을 약 1.4 s에 내린다. ESTOP만 즉시. 새 핀 값(GRID)이 박자에 적용되면 보통 속도로 돌아간다.
constexpr uint32_t RETURN_STEP_MS = 10;

constexpr uint32_t UPDATE_PERIOD_MS = 5;       // 서보 갱신 200 Hz
constexpr uint16_t SLEW_TICKS_PER_UPDATE = 4;  // SG90 부하 약 360도/s ≈ 778 tick/s → 5 ms당 약 4 tick
constexpr uint32_t COMM_TIMEOUT_MS = 500;

constexpr uint16_t TICK_PERIOD_MS_DEFAULT = 200;  // 핀 갱신 박자 (초당 5회)
constexpr uint16_t TICK_PERIOD_MS_MIN = 20;
constexpr uint16_t TICK_PERIOD_MS_MAX = 2000;
constexpr uint8_t JITTER_TICKS_MAX = 40;
constexpr uint8_t MAX_HAZARD = 3;

// 프로토콜
constexpr uint8_t SYNC0 = 0xAA;
constexpr uint8_t SYNC1 = 0x55;
constexpr uint8_t TYPE_GRID = 0x01;
constexpr uint8_t TYPE_RAW_TICK = 0x02;
constexpr uint8_t TYPE_HOME_ALL = 0x03;
constexpr uint8_t TYPE_SET_LIMIT = 0x04;
constexpr uint8_t TYPE_PING = 0x05;
constexpr uint8_t TYPE_SET_CONFIG = 0x06;
constexpr uint8_t TYPE_KEEPALIVE = 0x07;  // v3: 내용 없음, failsafe 타이머만 갱신
constexpr uint8_t FLAG_ESTOP = 0x01;
constexpr uint8_t MAX_PAYLOAD = 64;
constexpr uint8_t CONFIG_LEN_V2 = 4;  // levels, max_sim, (v2 진동 반주기 u16 — v3는 무시)
constexpr uint8_t CONFIG_LEN_V3 = 7;  // levels, max_sim, tick_period_ms u16, jitter tick ×3

// ─────────────────────────────────────────────────────────
// 순수 함수
// ─────────────────────────────────────────────────────────
// CRC-8 (poly 0x07, init 0x00) — protocol.py crc8()과 동일
inline uint8_t crc8(const uint8_t* data, size_t len) {
  uint8_t crc = 0x00;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

inline uint16_t clampTick(int32_t t) {
  if (t < TICK_HARD_MIN) return TICK_HARD_MIN;
  if (t > TICK_HARD_MAX) return TICK_HARD_MAX;
  return (uint16_t)t;
}

// 레벨 → tick. 0.5는 home에서 먼 쪽으로 반올림 — calibration.py tick_for_level()과 같은 값.
// home > max(역방향 장착)도 그대로 처리된다.
inline uint16_t levelTick(uint16_t homeTick, uint16_t maxTick, uint8_t level, uint8_t levels) {
  const int32_t top = (levels > 1) ? (int32_t)levels - 1 : 1;
  if ((int32_t)level > top) level = (uint8_t)top;
  const int32_t num = ((int32_t)maxTick - (int32_t)homeTick) * (int32_t)level;
  const int32_t mag = (2 * (num < 0 ? -num : num) + top) / (2 * top);
  return clampTick((int32_t)homeTick + (num < 0 ? -mag : mag));
}

struct ChannelCal {
  uint16_t homeTick;
  uint16_t maxTick;
};

using WriteFn = void (*)(uint8_t ch, uint16_t tick);
using SayFn = void (*)(const char* line);
using FrameHook = void (*)(uint8_t type, uint8_t seq, const uint8_t* payload, uint8_t len);

// ─────────────────────────────────────────────────────────
// 핵심 상태기계
// ─────────────────────────────────────────────────────────
class Core {
 public:
  // ── 설정 (SET_CONFIG / SET_LIMIT) ──
  ChannelCal cal[CELL_COUNT];
  uint8_t levels = 8;
  uint8_t maxSimultaneous = 8;
  uint16_t tickPeriodMs = TICK_PERIOD_MS_DEFAULT;
  uint8_t jitterTicks[MAX_HAZARD + 1] = {0, 2, 4, 6};

  // ── 상태 (시험용으로 공개) ──
  uint8_t level[CELL_COUNT];
  uint8_t hazard[CELL_COUNT];
  uint16_t targetTick[CELL_COUNT];   // 목표 위치 (진동 제외)
  uint16_t currentTick[CELL_COUNT];  // 명령 중인 위치
  uint16_t writtenTick[CELL_COUNT];  // PCA9685에 마지막으로 쓴 값 (중복 쓰기 방지)
  uint8_t pendLevel[CELL_COUNT];     // 다음 박자에 적용할 최신 값
  uint8_t pendHazard[CELL_COUNT];
  bool pending = false;
  bool estop = false;
  bool rawMode = false;  // RAW_TICK(캘리브레이션) 중 — 박자 적용·진동 멈춤
  bool failsafe = false;
  bool returning = false;  // 천천히 홈 복귀 중 (HOME_ALL·failsafe) — RETURN_STEP_MS마다 1 tick

  // ── 통계 (PING → PONG) ──
  uint32_t statFrames = 0, statCrcErr = 0, statResync = 0, statDropped = 0;
  uint32_t statBeats = 0, statApplied = 0, statSuperseded = 0, statFailsafe = 0;
  uint32_t statSayDropped = 0;  // 출력 버퍼가 차서 버린 상태 줄 (실기 .ino가 셈)

  void begin(WriteFn writeFn, SayFn sayFn, uint32_t now, FrameHook hook = nullptr) {
    writeFn_ = writeFn;
    sayFn_ = sayFn;
    hook_ = hook;
    for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
      cal[ch].homeTick = homeDefault(ch);
      cal[ch].maxTick = maxDefault(ch);
      level[ch] = hazard[ch] = pendLevel[ch] = pendHazard[ch] = 0;
      targetTick[ch] = currentTick[ch] = homeDefault(ch);
      writtenTick[ch] = 0xFFFF;  // 첫 쓰기를 강제
    }
    restartTimers(now);
  }

  // 기동 시 시간차 홈 쓰기(채널당 25 ms)가 failsafe·박자 타이머를 먹지 않게 다시 맞춘다.
  void restartTimers(uint32_t now) {
    lastRxMs_ = now;
    beatStartMs_ = now;
    lastUpdateMs_ = now;
  }

  // 수신 바이트 1개. 블로킹·동적할당 없음.
  void feed(uint8_t b, uint32_t now) {
    switch (rxState_) {
      case RX_S0:
        if (b == SYNC0) rxState_ = RX_S1;
        else statResync++;
        break;
      case RX_S1:
        if (b == SYNC1) rxState_ = RX_TYPE;
        else if (b == SYNC0) rxState_ = RX_S1;  // 0xAA 0xAA 0x55 대응
        else { statResync++; rxState_ = RX_S0; }
        break;
      case RX_TYPE: rxType_ = b; rxState_ = RX_SEQ; break;
      case RX_SEQ:  rxSeq_ = b;  rxState_ = RX_LEN; break;
      case RX_LEN:
        rxLen_ = b;
        if (rxLen_ > MAX_PAYLOAD) { statCrcErr++; rxState_ = RX_S0; break; }
        rxIdx_ = 0;
        rxState_ = (rxLen_ == 0) ? RX_CRC : RX_PAYLOAD;
        break;
      case RX_PAYLOAD:
        rxBuf_[rxIdx_++] = b;
        if (rxIdx_ >= rxLen_) rxState_ = RX_CRC;
        break;
      case RX_CRC: {
        uint8_t body[3 + MAX_PAYLOAD];
        body[0] = rxType_; body[1] = rxSeq_; body[2] = rxLen_;
        for (uint8_t i = 0; i < rxLen_; i++) body[3 + i] = rxBuf_[i];
        if (crc8(body, 3u + rxLen_) == b) handleFrame(rxType_, rxSeq_, rxBuf_, rxLen_, now);
        else statCrcErr++;
        rxState_ = RX_S0;
        break;
      }
    }
  }

  // 매 loop()마다 호출. failsafe → 박자 → 서보 갱신(5 ms마다).
  void poll(uint32_t now) {
    if (!failsafe && (uint32_t)(now - lastRxMs_) > COMM_TIMEOUT_MS) {
      failsafe = true;
      statFailsafe++;
      pending = false;
      rawMode = false;
      homeAll(false);  // 천천히 홈 — RETURN_STEP_MS마다 1 tick
      say("FAILSAFE no_comm");
    }
    const uint32_t el = now - beatStartMs_;
    if (el >= tickPeriodMs) {
      // 박자 위상 유지. 루프가 크게 밀렸으면(2박자 이상) 지금부터 다시 센다.
      beatStartMs_ = (el >= 2u * (uint32_t)tickPeriodMs) ? now : beatStartMs_ + tickPeriodMs;
      statBeats++;
      if (pending && !estop && !rawMode && !failsafe) applyPending();
    }
    if ((uint32_t)(now - lastUpdateMs_) >= UPDATE_PERIOD_MS) {
      lastUpdateMs_ = now;
      updateServos(now);
    }
  }

  // 박자 안의 1/4 구간 번호 0~3. 1·3이 진동 구간.
  uint8_t quarter(uint32_t now) const {
    const uint32_t el = now - beatStartMs_;
    if (el >= tickPeriodMs) return 0;
    return (uint8_t)((el * 4u) / tickPeriodMs);
  }

  void writeChannel(uint8_t ch, uint16_t tick) {
    if (writtenTick[ch] == tick) return;  // I2C 트래픽 절감
    writtenTick[ch] = tick;
    if (writeFn_) writeFn_(ch, tick);
  }

 private:
  enum RxState : uint8_t { RX_S0, RX_S1, RX_TYPE, RX_SEQ, RX_LEN, RX_PAYLOAD, RX_CRC };

  WriteFn writeFn_ = nullptr;
  SayFn sayFn_ = nullptr;
  FrameHook hook_ = nullptr;
  uint32_t lastRxMs_ = 0, beatStartMs_ = 0, lastUpdateMs_ = 0, lastReturnMs_ = 0;
  uint8_t lastSeq_ = 0;
  bool seqInit_ = false;
  RxState rxState_ = RX_S0;
  uint8_t rxType_ = 0, rxSeq_ = 0, rxLen_ = 0, rxIdx_ = 0;
  uint8_t rxBuf_[MAX_PAYLOAD];

  void say(const char* s) {
    if (sayFn_) sayFn_(s);
  }

  void sayf(const char* fmt, ...) {
    if (!sayFn_) return;
    char buf[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    sayFn_(buf);
  }

  uint16_t tickOf(uint8_t ch, uint8_t lv) const {
    return levelTick(cal[ch].homeTick, cal[ch].maxTick, lv, levels);
  }

  // immediate = ESTOP(즉시 점프). 아니면 천천히 복귀(returning) — 종료·통신 끊김 때.
  void homeAll(bool immediate) {
    for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
      level[ch] = 0;
      hazard[ch] = 0;
      targetTick[ch] = clampTick(cal[ch].homeTick);
      if (immediate) {
        currentTick[ch] = targetTick[ch];
        writeChannel(ch, currentTick[ch]);
      }
    }
    returning = !immediate;
  }

  void applyPending() {
    returning = false;  // 새 핀 값 — 보통 속도로
    for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
      level[ch] = pendLevel[ch];
      hazard[ch] = pendHazard[ch];
      targetTick[ch] = tickOf(ch, level[ch]);
    }
    pending = false;
    statApplied++;
  }

  // 진동 목표: 홈 쪽으로 a tick. 보정 범위를 벗어나면 반대(위)쪽으로. 범위 안으로 클램프.
  uint16_t jitterTarget(uint8_t ch, uint16_t base) const {
    const uint8_t hz = hazard[ch] > MAX_HAZARD ? MAX_HAZARD : hazard[ch];
    const int32_t a = jitterTicks[hz];
    if (a == 0) return base;
    const int32_t home = cal[ch].homeTick, mx = cal[ch].maxTick;
    const int32_t lo = home < mx ? home : mx;
    const int32_t hi = home < mx ? mx : home;
    const int32_t toHome = (home <= mx) ? -1 : +1;
    int32_t w = (int32_t)base + toHome * a;
    if (w < lo || w > hi) w = (int32_t)base - toHome * a;
    if (w < lo) w = lo;
    if (w > hi) w = hi;
    return clampTick(w);
  }

  void updateServos(uint32_t now) {
    int32_t maxStep = SLEW_TICKS_PER_UPDATE;
    if (returning) {  // 천천히 홈: RETURN_STEP_MS마다 1 tick
      if ((uint32_t)(now - lastReturnMs_) < RETURN_STEP_MS) return;
      lastReturnMs_ = now;
      maxStep = 1;
    }
    bool jitterOn = false;
    if (!estop && !rawMode && !failsafe && !returning) {
      const uint8_t q = quarter(now);
      jitterOn = (q == 1 || q == 3);
    }

    // 움직여야 할 채널 모으기. 우선순위: (1) 상승(장애물 접근 = 위험 신호), (2) 오차 큰 순.
    uint16_t want[CELL_COUNT];
    uint8_t cand[CELL_COUNT];
    uint16_t err[CELL_COUNT];
    bool rising[CELL_COUNT];
    uint8_t n = 0;
    for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
      uint16_t w = targetTick[ch];
      if (jitterOn && hazard[ch] > 0) w = jitterTarget(ch, w);
      want[ch] = w;
      const int32_t d = (int32_t)w - (int32_t)currentTick[ch];
      if (d == 0) continue;
      const bool upIsPlus = cal[ch].maxTick >= cal[ch].homeTick;  // 역방향 장착 고려
      cand[n] = ch;
      err[n] = (uint16_t)(d < 0 ? -d : d);
      rising[n] = upIsPlus ? (d > 0) : (d < 0);
      n++;
    }

    // 부분 선택 — 최대 maxSimultaneous개만 (전류 상한)
    uint8_t budget = maxSimultaneous ? maxSimultaneous : 1;
    if (budget > n) budget = n;
    bool used[CELL_COUNT] = {false};
    for (uint8_t pick = 0; pick < budget; pick++) {
      int16_t best = -1;
      for (uint8_t i = 0; i < n; i++) {
        if (used[i]) continue;
        if (best < 0) { best = i; continue; }
        if (rising[i] != rising[best]) { if (rising[i]) best = i; }
        else if (err[i] > err[best]) { best = i; }
      }
      if (best < 0) break;
      used[best] = true;
      const uint8_t ch = cand[best];
      int32_t step = (int32_t)want[ch] - (int32_t)currentTick[ch];
      if (step > maxStep) step = maxStep;
      if (step < -maxStep) step = -maxStep;
      currentTick[ch] = clampTick((int32_t)currentTick[ch] + step);
      writeChannel(ch, currentTick[ch]);
    }
  }

  void handleFrame(uint8_t type, uint8_t seq, const uint8_t* p, uint8_t len, uint32_t now) {
    statFrames++;
    lastRxMs_ = now;
    if (failsafe) {
      failsafe = false;
      say("COMM ok");
    }
    if (seqInit_) {  // 시퀀스 유실 감지 (패킷 손실률 측정용)
      const uint8_t expect = (uint8_t)(lastSeq_ + 1);
      if (seq != expect) statDropped += (uint8_t)(seq - expect);
    }
    lastSeq_ = seq;
    seqInit_ = true;
    if (hook_) hook_(type, seq, p, len);

    switch (type) {
      case TYPE_GRID: onGrid(p, len); break;
      case TYPE_RAW_TICK: onRawTick(p, len); break;
      case TYPE_HOME_ALL:  // 종료 시 젯슨이 보낸다 — 천천히 맨 아래로
        estop = false;
        rawMode = false;
        pending = false;
        homeAll(false);
        say("HOMED");
        break;
      case TYPE_SET_LIMIT: onSetLimit(p, len); break;
      case TYPE_PING:
        sayf("PONG v=%u f=%lu crc=%lu resync=%lu drop=%lu beats=%lu applied=%lu sup=%lu fs=%lu saydrop=%lu",
             (unsigned)FW_VERSION, (unsigned long)statFrames, (unsigned long)statCrcErr,
             (unsigned long)statResync, (unsigned long)statDropped, (unsigned long)statBeats,
             (unsigned long)statApplied, (unsigned long)statSuperseded,
             (unsigned long)statFailsafe, (unsigned long)statSayDropped);
        break;
      case TYPE_SET_CONFIG: onSetConfig(p, len, now); break;
      case TYPE_KEEPALIVE: break;
      default: sayf("ERR type=%u", (unsigned)type);
    }
  }

  void onGrid(const uint8_t* p, uint8_t len) {
    if (len != CELL_COUNT + 1) {
      sayf("ERR grid_len=%u", (unsigned)len);
      return;
    }
    if (p[CELL_COUNT] & FLAG_ESTOP) {
      const bool was = estop;
      estop = true;
      pending = false;
      rawMode = false;
      homeAll(true);  // 즉시 — 박자를 기다리지 않는다
      if (!was) say("ESTOP");
      return;
    }
    estop = false;
    rawMode = false;
    if (pending) statSuperseded++;  // 적용 전에 더 새 값이 왔다
    for (uint8_t ch = 0; ch < CELL_COUNT; ch++) {
      pendLevel[ch] = p[ch] & 0x0F;
      pendHazard[ch] = (p[ch] >> 4) & 0x0F;
    }
    pending = true;
  }

  void onRawTick(const uint8_t* p, uint8_t len) {
    if (len != 3) { say("ERR raw_len"); return; }
    const uint8_t ch = p[0];
    const uint16_t tick = (uint16_t)((p[1] << 8) | p[2]);
    if (ch >= CELL_COUNT) { say("ERR raw_ch"); return; }
    const uint16_t t = clampTick(tick);
    rawMode = true;
    returning = false;
    pending = false;
    for (uint8_t i = 0; i < CELL_COUNT; i++) hazard[i] = 0;
    // 캘리브레이션: 슬루 제한을 우회해 즉시 인가
    targetTick[ch] = t;
    currentTick[ch] = t;
    writeChannel(ch, t);
    sayf("TICK %u %u", (unsigned)ch, (unsigned)t);
  }

  void onSetLimit(const uint8_t* p, uint8_t len) {
    if (len != 5) { say("ERR lim_len"); return; }
    const uint8_t ch = p[0];
    const uint16_t hm = (uint16_t)((p[1] << 8) | p[2]);
    const uint16_t mx = (uint16_t)((p[3] << 8) | p[4]);
    if (ch >= CELL_COUNT) { say("ERR lim_ch"); return; }
    cal[ch].homeTick = clampTick(hm);
    cal[ch].maxTick = clampTick(mx);
    targetTick[ch] = tickOf(ch, level[ch]);
    sayf("LIMIT %u %u %u", (unsigned)ch, (unsigned)cal[ch].homeTick, (unsigned)cal[ch].maxTick);
  }

  void onSetConfig(const uint8_t* p, uint8_t len, uint32_t now) {
    if (len != CONFIG_LEN_V2 && len != CONFIG_LEN_V3) {
      sayf("ERR cfg_len=%u", (unsigned)len);
      return;
    }
    const uint8_t lv = p[0], ms = p[1];
    if (lv >= 2 && lv <= 16) levels = lv;
    if (ms >= 1 && ms <= CELL_COUNT) maxSimultaneous = ms;
    if (len == CONFIG_LEN_V3) {
      const uint16_t tp = (uint16_t)((p[2] << 8) | p[3]);
      if (tp >= TICK_PERIOD_MS_MIN && tp <= TICK_PERIOD_MS_MAX && tp != tickPeriodMs) {
        tickPeriodMs = tp;
        beatStartMs_ = now;  // 새 박자는 지금부터
      }
      for (uint8_t i = 0; i < MAX_HAZARD; i++) {
        if (p[4 + i] <= JITTER_TICKS_MAX) jitterTicks[1 + i] = p[4 + i];
      }
    }
    // v2 형식(4바이트)의 3·4번째 바이트는 v2의 진동 반주기다. v3는 박자로 진동 시점을 정하므로 무시.
    if (!rawMode) {
      for (uint8_t ch = 0; ch < CELL_COUNT; ch++) targetTick[ch] = tickOf(ch, level[ch]);
    }
    sayf("CFG lv=%u sim=%u tick=%u jit=%u,%u,%u%s", (unsigned)levels, (unsigned)maxSimultaneous,
         (unsigned)tickPeriodMs, (unsigned)jitterTicks[1], (unsigned)jitterTicks[2],
         (unsigned)jitterTicks[3], len == CONFIG_LEN_V2 ? " (v2 cfg)" : "");
  }
};

}  // namespace tv
