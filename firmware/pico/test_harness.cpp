// 펌웨어 PC 교차검증 하네스 — 실기 펌웨어와 같은 tactile_pico/tactile_core.h를 그대로 쓴다(복사본 없음).
//
// 빌드: g++ -O2 -std=c++17 -Wall -Wextra -o /tmp/fw_test firmware/pico/test_harness.cpp
//
//   fw_test < frames.bin            파이썬이 만든 바이트열 파싱 → FRAME 줄 + "STATS frames= crc_err= resync="
//   fw_test --crc HEX               CRC8
//   fw_test --ticks HOME MAX LEVELS 레벨별 tick (calibration.py tick_for_level과 비교)
//   fw_test --sim END_MS < events   가상 시간(1 ms 간격) 시뮬레이션. events = (u32 t_ms LE, u16 len LE, bytes)*
//                                   t_ms에 바이트를 넣고 poll(t_ms). 출력:
//                                     W t ch tick   PWM 쓰기
//                                     T t ch tick   목표 위치 변화(진동 제외) — 박자 적용 확인용
//                                     S t 메시지    상태 줄 (PONG, CFG, FAILSAFE …)
//                                     STATS ...     마지막 통계
// 시험 스크립트: tools/test_firmware.py

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tactile_pico/tactile_core.h"

static tv::Core core;
static uint32_t g_now = 0;

static void hookPrint(uint8_t type, uint8_t seq, const uint8_t* p, uint8_t len) {
  printf("FRAME type=%u seq=%u len=%u payload=", type, seq, len);
  for (uint8_t i = 0; i < len; i++) printf("%02x", p[i]);
  printf("\n");
  if (type == tv::TYPE_GRID && len == tv::CELL_COUNT + 1) {
    printf("  levels=");
    for (uint8_t i = 0; i < tv::CELL_COUNT; i++) printf("%u,", p[i] & 0x0F);
    printf("\n  hazards=");
    for (uint8_t i = 0; i < tv::CELL_COUNT; i++) printf("%u,", (p[i] >> 4) & 0x0F);
    printf("\n  flags=%u\n", p[tv::CELL_COUNT]);
  }
}

static void writeLog(uint8_t ch, uint16_t tick) { printf("W %u %u %u\n", g_now, ch, tick); }
static void sayLog(const char* s) { printf("S %u %s\n", g_now, s); }

static std::vector<uint8_t> readStdin() {
  std::vector<uint8_t> d;
  int c;
  while ((c = getchar()) != EOF) d.push_back((uint8_t)c);
  return d;
}

static int simulate(uint32_t endMs) {
  struct Ev { uint32_t t; size_t off, len; };
  const std::vector<uint8_t> d = readStdin();
  std::vector<Ev> ev;
  size_t i = 0;
  while (i + 6 <= d.size()) {
    const uint32_t t = (uint32_t)d[i] | ((uint32_t)d[i + 1] << 8) | ((uint32_t)d[i + 2] << 16) |
                       ((uint32_t)d[i + 3] << 24);
    const size_t n = (size_t)d[i + 4] | ((size_t)d[i + 5] << 8);
    i += 6;
    if (i + n > d.size()) {
      fprintf(stderr, "events: 잘린 레코드\n");
      return 2;
    }
    ev.push_back({t, i, n});
    i += n;
  }
  if (i != d.size()) {
    fprintf(stderr, "events: 남는 바이트 %zu\n", d.size() - i);
    return 2;
  }

  g_now = 0;
  core.begin(writeLog, sayLog, 0);
  for (uint8_t ch = 0; ch < tv::CELL_COUNT; ch++) core.writeChannel(ch, core.currentTick[ch]);
  core.restartTimers(0);

  // 목표 위치 변화(진동 제외)도 출력한다: "T t ch tick" — 박자 적용 시점을 정확히 보려고
  uint16_t prevTarget[tv::CELL_COUNT];
  for (uint8_t ch = 0; ch < tv::CELL_COUNT; ch++) prevTarget[ch] = core.targetTick[ch];

  size_t e = 0;
  for (uint32_t t = 0; t <= endMs; t++) {
    g_now = t;
    while (e < ev.size() && ev[e].t <= t) {
      for (size_t k = 0; k < ev[e].len; k++) core.feed(d[ev[e].off + k], t);
      e++;
    }
    core.poll(t);
    for (uint8_t ch = 0; ch < tv::CELL_COUNT; ch++) {
      if (core.targetTick[ch] != prevTarget[ch]) {
        prevTarget[ch] = core.targetTick[ch];
        printf("T %u %u %u\n", t, ch, prevTarget[ch]);
      }
    }
  }
  printf("STATS frames=%u crc_err=%u resync=%u drop=%u beats=%u applied=%u sup=%u fs=%u\n",
         core.statFrames, core.statCrcErr, core.statResync, core.statDropped, core.statBeats,
         core.statApplied, core.statSuperseded, core.statFailsafe);
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 2 && strcmp(argv[1], "--crc") == 0) {
    const char* hex = argv[2];
    std::vector<uint8_t> d;
    for (size_t i = 0; hex[i] && hex[i + 1]; i += 2) {
      unsigned v;
      if (sscanf(hex + i, "%2x", &v) != 1) return 2;
      d.push_back((uint8_t)v);
    }
    printf("%02x\n", tv::crc8(d.data(), d.size()));
    return 0;
  }
  if (argc > 4 && strcmp(argv[1], "--ticks") == 0) {
    const uint16_t home = (uint16_t)atoi(argv[2]), mx = (uint16_t)atoi(argv[3]);
    const uint8_t levels = (uint8_t)atoi(argv[4]);
    for (uint8_t lv = 0; lv < levels; lv++) printf("%u ", tv::levelTick(home, mx, lv, levels));
    printf("\n");
    return 0;
  }
  if (argc > 2 && strcmp(argv[1], "--sim") == 0) return simulate((uint32_t)strtoul(argv[2], nullptr, 10));
  if (argc > 1 && strcmp(argv[1], "--map") == 0) {  // 칸 → 보드(0=0x40, 1=0x41) 채널
    for (uint8_t c = 0; c < tv::CELL_COUNT; c++)
      printf("%u %u %u\n", c, tv::boardOf(c), tv::CELL_TO_CHANNEL[c]);
    printf("MAP %s\n", tv::CHANNEL_MAP_NAME);
    return 0;
  }

  // 기본: 바이트열 파싱 (하드웨어 콜백 없음 — 프레임만 출력)
  core.begin(nullptr, nullptr, 0, hookPrint);
  for (uint8_t b : readStdin()) core.feed(b, 0);
  printf("STATS frames=%u crc_err=%u resync=%u\n", core.statFrames, core.statCrcErr, core.statResync);
  return 0;
}
