# 사고 기록: 외부 디렉토리 파일 덮어쓰기 (2026-09-22)

## 무슨 일이 있었나

이 프로젝트를 구축하는 과정에서 위험도 검출용 YOLO 엔진을 빌드했다. 그때
`ultralytics.YOLO.export()` 를 **다른 프로젝트 폴더에 있는 `.pt` 파일에 직접**
호출했다.

```python
# 사고를 낸 코드
m = YOLO('/home/june/sensory_substitution/Depth-Anything-V2/yolov8n.pt')
m.export(format='engine', half=True, imgsz=(288,512), device=0)
```

`export()` 는 **입력 `.pt` 가 있는 디렉토리에 결과를 쓴다.** 그래서 그 폴더의
기존 파일 2개가 경고 없이 덮어써졌다.

| 파일 | 원본 | 덮어쓴 값 |
|---|---|---|
| `yolov8n.onnx` | 6,463,259 B (FP16, 640×640), 07-27 22:40 | 13 MB (FP32, 288×512) |
| `yolov8n.engine` | ~9,020,000 B 추정 (8.6M 표시), 07-27 22:47 | 9.1 MB (288×512) |

부수적으로 `metric_depth/**/__pycache__/` 도 생성되었다 (외부 모듈 import 부작용).

## 복구 결과

| 파일 | 상태 |
|---|---|
| `yolov8n.pt` | 손대지 않음 (원본 유지) — 덕분에 재빌드가 가능했다 |
| `yolov8n.onnx` | **바이트 크기 일치 복구** (6,463,259 B). `export(format='onnx', half=True, imgsz=640)` 로 재현 |
| `yolov8n.engine` | **기능 동등 복구** (9,541,912 B). 바이트 일치는 불가능 |
| `__pycache__` | 삭제 (원상 복구) |
| `depth_anything_v2_vits.*` | 처음부터 무영향 (07-27 타임스탬프 유지 확인) |

### 엔진 바이트 일치가 불가능한 이유

TensorRT 엔진은 재현 가능한 산출물이 아니다. 빌더가 실행 시점에 커널 타이밍을
측정해 전술(tactic)을 선택하므로, 같은 ONNX·같은 플래그라도 워크스페이스 크기,
TRT 버전, 장비 부하에 따라 다른 커널이 선택되고 크기가 달라진다.

재빌드 2회 시도 결과:

| 방식 | 바이트 |
|---|---|
| 원본 (기록으로만 남음) | ~9,020,000 추정 |
| TRT Python API (FP16 ONNX에서) | 9,498,716 |
| `ultralytics export(format='engine')` | 9,541,912 ← 채택 |

`ultralytics` 빌드를 채택한 이유는 COCO 클래스 이름 메타데이터를 포함하기
때문이다. 원시 TRT API 빌드는 메타데이터가 없어 클래스가 `class0`, `class2` 로
표시된다.

### 복구 검증

```
클래스 메타데이터  80개, 0=person 2=car 5=bus 74=clock
demo01.jpg 검출   .pt    {person:13, car:6, bus:1, truck:2}
                  engine {person:13, car:6, bus:1, truck:2}   완전 일치
반복 추론          15회 오류 없음
ONNX 구조 검사     통과, 입력 [1,3,640,640] 출력 [1,84,8400]
```

참고: 원본 폴더의 어떤 코드도 `yolov8n.engine` 을 참조하지 않았다
(`src/` 는 `checkpoints/yolo26n.engine` 사용). 실험용 잔여 파일이었다.

## 재발 방지 조치

1. **모델 정의 코드를 `vendor/` 로 복사했다.** 외부 트리를 `sys.path` 에 넣고
   import 하면 그쪽에 `__pycache__` 가 생성된다. 이제 외부 모듈을 import하지 않는다.

2. **체크포인트를 `checkpoints/` 로 옮겼다.** 프로젝트가 자립적이다.

3. **`tools/build_yolo.py` 를 만들어 출력을 격리했다.** `.pt` 를
   `tempfile.TemporaryDirectory()` 로 복사한 뒤 거기서 내보내고, 완성된 엔진만
   `models/` 로 옮긴다. 원본 디렉토리는 읽기만 한다.

4. **`tools/build_engine.py` 의 외부 경로 상수를 제거했다.**
   `VENDOR_ROOT = REPO / "vendor"`, `CKPT_DIR = REPO / "checkpoints"`.

검증: `grep -rn "sensory_substitution" --include="*.py" --include="*.yaml" .` 결과
코드·설정에 외부 경로 참조 0건. (README의 venv 실행 경로 1건은 읽기 전용 안내)

## 남은 원칙

- 프로젝트는 자기 디렉토리 밖에 파일을 쓰지 않는다.
- 외부 파일을 입력으로 받는 도구는 출력 위치를 명시적으로 통제한다.
  라이브러리의 기본 출력 위치를 신뢰하지 않는다.
- 되돌릴 수 없는 작업(덮어쓰기, 삭제) 전에는 대상이 프로젝트 밖인지 확인한다.
