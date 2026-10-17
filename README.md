# Tactile Vision — 시각-촉각 감각대체 장치

단안 카메라의 **절대 거리(metric depth)** 를 양손 손바닥의 24점 촉각 배열로 변환해
시각장애인의 독립 보행을 보조하는 시스템.

```
USB 카메라 Logitech C920 (1280×720, 자동초점 끔·초점 0)
    │  종횡비 정합 크롭 1280×698 → 924×504
    ▼
Jetson Orin Nano Super 8GB (25W)                       2026-10-03 실측
    ├─ Places365 실내/실외 판별 (초당 2회)               판별하는 프레임 +22 ms
    │    → 실내(Hypersim)·실외(VKITTI) 모델과 d_far 자동 전환 (애매해도 모델 1개)
    ├─ Depth Anything V2 Metric (ViT-S, TensorRT FP16)   48~50 ms
    ├─ 6×4 셀 풀링 (GPU, 하위 10분위 = 최근접 표면)        (위에 포함)
    ├─ 비대칭 시간 필터 + 8단계 로그 양자화               <0.6 ms
    └─ YOLO26n 위험도 검출(COCO) → 위험도 1/2/3          10~15 ms
    │  계산될 때마다 전송 (약 15 Hz) — 바이너리 31 B + CRC8
    ▼
RP2040 (Pico) 펌웨어 v3
    ├─ 받은 값은 보관만, 박자(초당 5회)마다 최신 값 하나를 핀에 적용
    ├─ 박자 사이 2번(초당 10회) 위험도 진동 1/2/3도 (맨 아래 핀은 위로)
    ├─ 채널별 캘리브레이션 + 슬루 제한 + 동시 구동 제한 (전류 보호)
    ├─ 종료(HOME_ALL)·통신 끊김(0.5 s) → 모든 핀을 범위 맨 아래로 천천히 (1 tick/10 ms, 끝→끝 약 1.4 s)
    └─ I2C 400 kHz → PCA9685 ×2 → SG90 ×24
    │
    ▼
랙&피니언 Z축 핀 — 스트로크 20.9 mm, 8단계 (간격 2.99 mm)
```

## `ground-scale` 가지 (2026-10-17) — 실외 땅 기준 거리 환산

거리 사진 120장(실외 산책로·주차장 1~8 m, 실내 거실·주방 0.5~4 m, 렌즈 높이 0.70 m, 줄자 기준)으로
두 metric 모델을 검증한 결과에 따라 바꾼 가지다. `main`은 그대로다.

- **실외**: metric 실외 모델(VKITTI2)은 보행 높이 사진에서 3~6배 멀게 봤다(실외 MAPE 519%). 학습 카메라 높이
  1.65 m와 화각 차이로 계산한 3.5배와 맞는다. 대신 **상대깊이 모델 + 땅 기준 환산**(`ground.py`)을 쓴다.
  화면 아래 땅 줄의 거리를 카메라 높이·기울기·초점거리로 계산해 프레임마다 배율을 정한다. 출력은 그대로 미터다.
  실외 MAPE 21.1%(실내 모델 − 0.44 m를 썼을 때 38.0%). 상자 거리로 맞춘 값은 없다.
- **실내**: 실내 모델(Hypersim)은 거리와 관계없이 약 0.44 m 멀게 봤다 → `depth.indoor_offset_m: -0.44`
  (MAPE 34% → 9.5%, 장소 교차).
- 실내/실외 판별·d_far(4 / 8 m)·매핑·필터·펌웨어는 바뀌지 않았다.
- 한계: 기울기 1°·높이 5 cm가 틀리면 실외 MAPE가 약 5%p씩 나빠진다(`ground` 섹션을 착용 상태로 맞출 것).
  땅이 화면 아래에 보여야 한다. 걸을 때의 흔들림(배율 시간 평활)은 검증하지 않았다.
- 시험: `tools/test_ground.py`(41항목, CPU) + 기존 199항목.

## 설계 원칙

이 시스템의 모든 수치는 **STL 실측**과 **손바닥 지각 한계**에서 유도된다.
자세한 근거는 [docs/hardware_measured.md](docs/hardware_measured.md).

### 1. 절대 거리 + 로그 매핑 (상대 깊이 아님)

프레임별 min-max 정규화를 쓰면 같은 핀 높이가 장면마다 다른 거리를 뜻한다.
실측 비교 — 같은 레벨 4가 의미하는 거리:

| 장면 | metric+log | relative_ema |
|---|---|---|
| 복도 (0.5~10m) | 1.74 m | 4.63 m |
| 벽 앞 (1.8~2.2m) | 1.90 m | 1.97 m |
| 로비 (3~15m) | (해당 레벨 없음) | 11.35 m |
| **장면간 변동** | **1.09배** | **5.75배** |

상대 매핑은 사용자가 "이 높이 = 이 거리"를 학습할 수 없게 만들고, 깊이 분산이
작은 장면에서는 모델 노이즈를 손바닥으로 증폭 전달한다.

로그 매핑을 쓰는 이유는 8단계밖에 없기 때문이다. 선형은 far bound 밖이 전부
포화되어 원거리 정보를 잃는다.

| 레벨 | 거리 구간 | 구간 폭 | 핀 높이 |
|---|---|---|---|
| 7 | 0.50 ~ 0.71 m | 0.21 m | 20.9 mm |
| 6 | 0.71 ~ 1.00 m | 0.29 m | 17.9 mm |
| 5 | 1.00 ~ 1.41 m | 0.41 m | 14.9 mm |
| 4 | 1.41 ~ 2.00 m | 0.59 m | 11.9 mm |
| 3 | 2.00 ~ 2.83 m | 0.83 m | 9.0 mm |
| 2 | 2.83 ~ 4.00 m | 1.17 m | 6.0 mm |
| 1 | 4.00 ~ 5.66 m | 1.66 m | 3.0 mm |
| 0 | 5.66 ~ 8.00 m | 2.34 m | 0.0 mm |

충돌 위험 구간(0.5~1m)에 2단계를 배정하면서 8m까지 구분을 유지한다.
위 표는 실외 d_far 8 m 기준이다. 실내는 d_far 4 m (`python run.py --show-mapping`).

### 2. 단계 수는 지각 한계가 정한다

스트로크 20.9 mm ÷ 손바닥 높이 JND 2.5 mm → 최대 9단계. **8단계 채택**(간격 2.99 mm).

0~255를 그대로 보내면 지각 한계의 30배 해상도를 명령하는 것이고, 결과는
사람이 느끼지도 못하는 서보 미세 이동 — 떨림, 전류 낭비, 기어 마모다.

### 3. 시간 필터는 방향 비대칭

핀이 늦게 올라가면 충돌이지만, 늦게 내려가는 것은 안전하다.

| 필터 | 정지장면 서보 이동 | 접근 검출 지연 |
|---|---|---|
| 없음 | 442.4 회/분/서보 | — |
| 대칭 | 238.1 회/분/서보 | 24.7 cm |
| **비대칭 (채택)** | **0.15 회/분/서보** | **9.7 cm** |

급접근 시 1프레임에 다단계 즉시 상승, 급후퇴 시 1단계씩만 하강한다.

### 4. 셀 대표값은 평균이 아니라 하위 분위수

평균을 쓰면 기둥·의자다리 같은 얇은 장애물이 배경에 묻힌다.
같은 장면 실측: 평균 7.92 m vs 10분위 4.98 m — **2.9 m 차이**.

### 5. 병목은 서보다 — 그래서 박자는 Pico가 정한다

2026-10-03 실측 (실카메라 60 s, YOLO26n·Places365 포함, Pico 없이 — `logs/bench_2026-10-03/`):

| 구간 | 시간 |
|---|---|
| 깊이 924×504 (모델 1개) | 48.6 ms |
| 위험도 검출 (YOLO26n) | 14.7 ms |
| 장면 판별 (초당 2회, 그 프레임만) | +22.5 ms |
| 프레임 전체 | 중앙 64.1 ms / p95 85.9 ms → **초당 15.3 프레임** |
| 서보 1단계 이동 *(산정치)* | 59.4 ms |
| 서보 전체 스트로크 *(산정치)* | **416 ms** |

젯슨은 결과가 나올 때마다(약 15 Hz) 보내고, Pico가 자기 박자(`transport.pin_rate_hz`, 기본
초당 5회)마다 그 순간의 최신 값 하나만 핀에 적용한다. 박자 200 ms 안에 새 결과가 약 3개
오므로 계산이 조금 들쭉날쭉해도 박자는 일정하고, 서보는 박자 안에 약 3단계까지 움직인다.
박자 사이의 50~100·150~200 ms에는 위험 셀을 1/2/3도 비켜 놓는다(초당 10회 진동).

1단계 변화의 종단 지연은 평균 약 258 ms(박자 대기 최악 시 +100 ms), 전체 스트로크 변화는
약 615 ms다 *(산정치, `tools/benchmark.py`)*. Pico의 부담은 작다: 수신 약 470 B/s, 최악 조건
(24핀 끝↔끝 + 6칸 진동)에서 I2C 쓰기 약 1,500회/s ≈ I2C 점유 20% *(산정치, `tools/test_firmware.py`)*.

## 설치

### 모델 파일 (저장소에 없음)

가중치·엔진은 용량과 라이선스 때문에 올리지 않았다(`.gitignore`). 아래 위치에 받아 두고 엔진을 직접 빌드한다.
TensorRT 엔진은 빌드한 장치(Jetson Orin Nano, TensorRT 10.16)에서만 쓸 수 있다.

| 파일 | 위치 | 출처 |
|---|---|---|
| `depth_anything_v2_metric_hypersim_vits.pth` (실내) | `checkpoints/` | [Hugging Face depth-anything/Depth-Anything-V2-Metric-Hypersim-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small) |
| `depth_anything_v2_metric_vkitti_vits.pth` (실외) | `checkpoints/` | [Hugging Face depth-anything/Depth-Anything-V2-Metric-VKITTI-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small) |
| `depth_anything_v2_vits.pth` (실외 상대깊이, `ground-scale` 가지) | `checkpoints/` | [Hugging Face depth-anything/Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) |
| `yolo26n.pt` (위험도) | `checkpoints/` | Ultralytics YOLO26n 공식 가중치 |
| `resnet18_places365.pth.tar`, `categories_places365.txt`, `IO_places365.txt` (실내/실외) | `checkpoints/places365/` | [CSAILVision/places365](https://github.com/CSAILVision/places365) (`src/tactile_vision/scene.py` 머리말에 주소) |
| `*.onnx`, `*.engine` | `models/` | 아래 빌드 명령이 만든다 |

`vendor/depth_anything_v2/`는 [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)의 metric 모델 정의를
복사한 것이다 (Apache-2.0, `vendor/depth_anything_v2/LICENSE`).

### 설치 명령

```bash
# 의존성 (Jetson에 이미 있는 venv 재사용)
VENV=/home/june/sensory_substitution/venv
$VENV/bin/pip install -r requirements.txt

# metric 깊이 엔진 빌드 (각 약 8분) — 기본 924×504
$VENV/bin/python tools/build_engine.py --spec wide_hi           # 실내 (Hypersim)
$VENV/bin/python tools/build_engine.py --spec wide_hi --outdoor # 실외 (VKITTI, main 기본)
$VENV/bin/python tools/build_engine.py --spec wide_hi --relative # 실외 상대깊이 (ground-scale 가지 기본)
# 위험도 검출 엔진 (약 8분, ultralytics가 패키지를 자동 설치하지 않게)
YOLO_AUTOINSTALL=false $VENV/bin/python tools/build_yolo.py --weights checkpoints/yolo26n.pt --imgsz 512 288

# Pico 펌웨어 컴파일 도구 (사용자 영역, 약 2 GB)
tools/setup_firmware_toolchain.sh

# 시리얼 권한 (최초 1회) — 2026-10-04 이 젯슨에 적용됨
sudo usermod -aG dialout $USER                       # 다음 로그인부터
sudo cp tools/60-tactile-pico.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules
                                                      # 재로그인 없이 바로 + ModemManager 제외 (Pico를 다시 꽂으면 적용)
```

Pico는 젯슨의 **USB-A 포트**에 꽂는다. 젯슨 본체의 USB-C 포트는 장치 모드(젯슨을 PC에 연결할 때 쓰는
포트)라 Pico가 인식되지 않는다.

## 사용

```bash
# 설계 수치 확인 (하드웨어 없이)
python run.py --show-mapping

# 실행
python run.py                        # 카메라 + Pico
python run.py --debug                # 시각화 창 포함
python run.py --no-hw                # Pico 없이
python run.py --source clip.mp4      # 녹화 영상으로 재현 실험

# 실내/실외 (기본: 자동 전환)
python run.py --scene indoor         # 실내 모델 + d_far 4 m 고정
python run.py --scene outdoor        # 실외 모델 + d_far 8 m 고정
python run.py --scene off            # 예전 방식 (실내 모델 + mapping.far_m)

# 실험 조건 전환 (코드 수정 없음)
python run.py --mapping log          # 제안
python run.py --mapping relative_ema # 기저선
python run.py --mapping linear       # 절제
```

### 핀 캘리브레이션 (조립 후 필수)

3D 프린트 공차로 24개 핀 높이가 다르다. 채널마다 home/max tick을 실측한다.
지금 값(2026-10-05): 24칸 모두 home 372(핀 맨 아래) / max 232(핀 맨 위) — 서보를 역방향으로 조립해 tick이 작을수록 핀이 위.
Pico 기동 기본값(`tactile_core.h` HOME/MAX_TICK_DEFAULT)과 젯슨 기본값(`calibration.py` DEVICE_*)도 같은 값이다.
범위를 바꾸면 세 곳을 함께 바꿀 것 (`tools/test_firmware.py`가 일치를 확인). 서보 시험 도구들은 이 파일의 범위·방향을 그대로 쓴다.

```bash
python tools/calibrate_pins.py
# h = 현재 위치를 home(레벨0)으로   m = max(레벨7)으로
# n/p = 채널 이동   t = 전체 스윕   space = 전체 홈(천천히)   s = 저장
```

**Arduino Uno로 서보 1개씩 측정하기 (2026-10-04)** — `firmware/calibration/uno_servo_sweep/`
서보 신호선을 D7에 꽂고(전원은 외부 5 V, GND 공통), `tools/upload_uno_sweep.sh [포트]`로 올리면
"최저점 → 최대 → 최저 → 1 s 쉼"을 반복한다. 기본은 시작 232 tick(≈60°), 범위 140 tick(≈64.8°, 9.0 mm).
D13 LED는 최대로 보내는 순간 켜지고 최저로 보내는 순간 꺼져 슬로모션 영상의 명령 기준이 된다.
시리얼 모니터(115200)에서 `?` 도움말, `s/w/h/p` 시작·범위·유지·쉼, `m 1` 슬루 모드(펌웨어와 같은 5 ms당 4 tick),
`t N`·`+`·`-` 조그(home/max tick 찾기), `r` 다시 시작. 모든 값은 82~533 tick(400~2600 µs)로 제한된다.

### 검증 및 벤치마크

```bash
python tools/test_mapping.py     # 매핑·필터 설계 주장 검증 (13개 항목)
python tools/test_protocol.py    # 프로토콜 신뢰성 (CRC, 재동기, 범위)
python tools/test_scene.py       # 실내/실외 판별기·판정 규칙
python tools/test_switch.py      # 자동 전환 (모델 선택, d_far 보간, 설정). --with-model: 실제 엔진
python tools/test_firmware.py    # Pico 펌웨어 v3 (박자 적용, 진동, failsafe, 설정, Python↔펌웨어 일치)
python tools/servo_updown_test.py  # 실물 배선 점검: 추론 없이 24칸 아래↔위 (I2C 0x40/0x41 응답도 보고)
python tools/servo_slow_updown.py  # 실물 동작 점검: 24개 모터가 함께 천천히 위아래 (범위 제한, 1 tick 단위 연속)
python tools/servo_wave_test.py    # 실물 순서 점검: 핀이 PCA 채널 순서대로 하나씩 파도처럼 (--order cell|column)
python tools/benchmark.py        # 단계별 지연 분해 + 종단 지연 예산
```

## 펌웨어 (v3)

```
firmware/pico/tactile_pico/tactile_pico.ino   하드웨어 연결 (PCA9685, USB 시리얼)
firmware/pico/tactile_pico/tactile_core.h     동작 규칙 전부 — PC 시험 하네스와 같은 코드
firmware/pico/legacy/tactile_pico_v2/         이전 펌웨어 (받은 값을 바로 적용)
배선: GP10=SDA, GP7=SCL (둘 다 I2C1 → 펌웨어는 Wire1) / PCA9685 0x40 = 1~2행, 0x41 = 3~4행 (A0 납땜)
      칸 → 채널 (보드마다 왼쪽 위부터 세로 2칸씩, tactile_core.h CELL_TO_CHANNEL):
        위 줄  0 2 4 6 8 10    (채널 0~5 = 왼손 열, 6~11 = 오른손 열)
        아래줄 1 3 5 7 9 11
      외부 5V → PCA9685 V+ (Pico 3V3와 분리)
```

```bash
tools/build_firmware.sh                    # 컴파일 → firmware/pico/build/tactile_pico.ino.uf2
# 업로드 ① 처음엔 이 방법: Pico BOOTSEL 버튼을 누른 채 USB 연결 → RPI-RP2 드라이브에 .uf2 복사
#          (MicroPython이 들어 있는 보드는 REPL에서 machine.bootloader() 로도 RPI-RP2 모드가 된다)
# 업로드 ② v3가 이미 있으면: tools/build_firmware.sh --upload /dev/ttyACM0  (위 시리얼 권한 필요)
```

2026-10-04 이 장치의 Pico(USB-C RP2040 보드)에 v3를 올리고 통신을 확인했다(PONG v=3, 설정 응답 일치).

의존: arduino-pico `rp2040:rp2040@6.2.0`, Adafruit PWM Servo Driver Library 3.0.3,
Adafruit BusIO 1.17.4 (`tools/setup_firmware_toolchain.sh`가 버전 고정 설치). 핀 갱신 박자와
진동 진폭은 실행 시 젯슨이 SET_CONFIG로 보내므로 바꿀 때 펌웨어를 다시 올릴 필요가 없다.

첫 구동 전 확인:
- `TICK_HARD_MIN/MAX`(82/533 = 400/2600µs)가 기구 한계 안인지
- 벅 컨버터 정격이 `safety.max_simultaneous_moves × 0.25 A` 이상인지
- 1도 진동(약 10 µs)이 SG90 데드밴드에 걸려 실제로 안 움직이는지 (`hazard.jitter_deg`로 조정)

## 구조

```
config/default.yaml            모든 튜닝 파라미터
src/tactile_vision/
  geometry.py                  STL 실측 상수 (단일 진실 공급원)
  config.py                    설정 로딩 + 물리/지각 제약 검증
  camera.py                    스레드 캡처, 종횡비 정합 크롭
  depth.py                     TRT metric 추론 + GPU 셀 풀링 (+ 실내 보정, 땅 기준 환산 연결)
  ground.py                    땅 기준 거리 환산 (상대깊이 → 미터, 카메라 높이·기울기·초점거리)
  mapping.py                   4가지 매핑 모드 + 비대칭 필터
  scene.py                     Places365 실내/실외 판별 + 판정 게이트
  switching.py                 자동 전환 정책 (모델 선택, d_far 보간)
  v4l2ctl.py                   카메라 자동초점 끄기·초점 고정
  hazard.py                    COCO 위험도(YOLO26n) → 셀별 위험도 1~3
  protocol.py                  바이너리 프레임 + CRC8 (GRID, SET_CONFIG v3, KEEPALIVE …)
  transport.py                 시리얼, 펌웨어 버전 판단, 권한 진단
  calibration.py               24채널 캘리브레이션 테이블
  pipeline.py                  단계별 계측 내장 파이프라인
  viz.py                       디버그 시각화
tools/                         빌드·캘리브레이션·검증·벤치마크, 펌웨어 도구 설치·빌드
firmware/pico/                 Pico 펌웨어 v3 (tactile_pico/) + PC 시험 하네스 + 이전 v2(legacy/)
docs/hardware_measured.md      STL 실측 제원
docs/experiment_design.md      논문 실험 설계
```

## 알려진 한계

- d_far(실내 4 m / 실외 8 m) 밖은 레벨 0으로 포화 (적응형 far bound는 옵션으로 구현, 기본 비활성)
- 실내·실외 모델의 거리 스케일이 달라(같은 장면에서 약 2.4~2.6배) 전환 순간 핀 높이가 바뀐다.
  d_far 0.7 s 보간과 비대칭 필터가 완화하지만 스케일 차이 자체는 남는다
- 배열 세로 스팬 120 mm가 손바닥 길이(~100 mm) 초과 → 행별 감도 불균일
- 진동은 초당 10회(박자의 2배), 진폭 1/2/3도 — 촉각 최적 대역(20~50 Hz)보다 느리고, 1도는 SG90
  데드밴드(약 10 µs, 데이터시트 명목) 근처라 실제로 움직이지 않을 수 있다 (실기 미확인)
- 1단계 변화 종단 지연 평균 약 0.26 s, 전체 스트로크 약 0.6 s *(산정치)* → 빠른 보행에서 부족할 수 있음
- 펌웨어 v3는 PC 시뮬레이션(`tools/test_firmware.py`)과 실기(2026-10-05, 카메라 → Pico → 24핀 실전 구동)에서 확인.
  1도 진동이 실제로 움직이는지, 24개 동시 구동 시 전류는 아직 측정하지 않았다
- 단안 metric 깊이의 절대 정확도를 RGB-D 기준으로 검증하지 않았음

## 라이선스

MIT (`LICENSE`). 단, `vendor/depth_anything_v2/`는 원본 Depth-Anything-V2의 Apache-2.0
(`vendor/depth_anything_v2/LICENSE`)을 따른다. 모델 가중치는 저장소에 없으며 각 배포처의 라이선스를 따른다.
