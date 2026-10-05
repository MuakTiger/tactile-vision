"""
실내/실외 자동 판별.

  SceneClassifier  카메라 프레임 → P(실외). Places365 ResNet18 + 공식 실내/실외 라벨.
  SceneGate        P(실외) 시계열 → 확정 판정(indoor | outdoor). 문턱에서의 요동을 막는다.

P(실외) 계산
  Places365는 사진을 365개 장면(복도, 거실, 주차장, 거리 …) 중 하나로 분류한다.
  공식 IO_places365.txt가 장면마다 실내(1)/실외(2)를 표시해 두었으므로

      P(실외) = Σ_장면 softmax(장면) × [장면이 실외]

  로 학습 없이 확률을 얻는다. 공식 데모는 상위 10개 장면의 다수결을 쓰지만, 확률
  가중합은 애매한 장면에서 0/1로 튀지 않고 연속적으로 움직여 게이트 입력으로 낫다.

전처리
  프레임 전체를 224×224로 줄인다(종횡비 무시). 공식 unified 데모와 같고, Places365
  학습 이미지도 원본을 256×256으로 줄인 것이다. 정사각 크롭은 좌우 44%를 버려
  화면 가장자리의 하늘·건물 같은 단서를 잃는다.

가중치 출처
  http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
  (CSAILVision/places365, MIT 라이선스). 공식 배포가 HTTP뿐이라
    - sha256을 아래 상수와 대조하고
    - torch.load(weights_only=True)로 연다 — pickle 안의 임의 코드 실행을 막는다.
  2026-09-28 검증: 독립 HTTPS 변환본(litert-community/Places365-ResNet18-LiteRT)의
  공개 결과와 top-3 일치, 확률 차 1.1%p 이내.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "resnet18_places365.pth.tar"
CATEGORIES_FILE = "categories_places365.txt"
IO_FILE = "IO_places365.txt"
WEIGHTS_SHA256 = "2f4759217d470da2b803f8f66cd4488a066406b555a5fb95ee9a4663f9f05588"
N_CLASSES = 365
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labels(ckpt_dir: Path) -> tuple[list[str], np.ndarray]:
    """(장면 이름 365개, 실외 여부 bool 365개). 두 파일의 순서가 같은지 검증한다."""
    cats = [ln.split()[0] for ln in (ckpt_dir / CATEGORIES_FILE).read_text().splitlines()
            if ln.strip()]
    io = [ln.split() for ln in (ckpt_dir / IO_FILE).read_text().splitlines() if ln.strip()]
    if len(cats) != N_CLASSES or len(io) != N_CLASSES:
        raise RuntimeError(f"라벨 수가 {N_CLASSES}이 아님: categories {len(cats)}, IO {len(io)}")
    if [row[0] for row in io] != cats:
        raise RuntimeError("categories와 IO 파일의 장면 순서가 다름")
    codes = {row[-1] for row in io}
    if not codes <= {"1", "2"}:
        raise RuntimeError(f"IO 라벨은 1(실내)/2(실외)만 있어야 함: {sorted(codes)}")
    names = [c[3:] for c in cats]  # '/a/airfield' → 'airfield', '/b/bazaar/indoor' → 'bazaar/indoor'
    outdoor = np.array([row[-1] == "2" for row in io], dtype=bool)
    return names, outdoor


@dataclass
class SceneResult:
    p_outdoor: float  # 0~1
    top: list[tuple[str, float, bool]]  # 상위 장면 (이름, 확률, 실외 여부)
    ms: float  # 전처리~결과까지 소요 시간


class SceneClassifier:
    """BGR uint8 프레임 → SceneResult. 2~5 Hz로 부르면 충분하다(장면은 천천히 바뀐다)."""

    def __init__(self, ckpt_dir: Path, device: str = "cuda", fp16: bool = True,
                 topk: int = 5, verify: bool = True) -> None:
        ckpt_dir = Path(ckpt_dir)
        wpath = ckpt_dir / WEIGHTS_FILE
        for p in (wpath, ckpt_dir / CATEGORIES_FILE, ckpt_dir / IO_FILE):
            if not p.exists():
                raise FileNotFoundError(
                    f"Places365 파일 없음: {p}\n"
                    "  가중치: http://places2.csail.mit.edu/models_places365/" + WEIGHTS_FILE +
                    "\n  라벨:   https://raw.githubusercontent.com/CSAILVision/places365/"
                    "master/{categories,IO}_places365.txt"
                )
        if verify:
            got = _sha256(wpath)
            if got != WEIGHTS_SHA256:
                raise RuntimeError(
                    f"가중치 sha256 불일치: {got}\n  기대값 {WEIGHTS_SHA256}\n"
                    "  파일이 손상되었거나 다른 파일 — 다시 받을 것"
                )

        self.names, outdoor = load_labels(ckpt_dir)
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.topk = topk

        ckpt = torch.load(wpath, map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        net = torchvision.models.resnet18(num_classes=N_CLASSES)  # weights=None: 다운로드 없음
        net.load_state_dict(state, strict=True)
        self.net = net.eval().to(device=device, dtype=self.dtype,
                                  memory_format=torch.channels_last)

        self._outdoor = torch.tensor(outdoor, dtype=torch.float32, device=device)
        self._outdoor_np = outdoor
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self._bgr2rgb = torch.tensor([2, 1, 0], device=device)
        logger.info("장면 판별기 로드: %s (%s, 실내 %d / 실외 %d 장면)", WEIGHTS_FILE,
                    "fp16" if fp16 else "fp32", int((~outdoor).sum()), int(outdoor.sum()))

    @torch.inference_mode()
    def classify(self, bgr: np.ndarray) -> SceneResult:
        t0 = time.perf_counter()
        small = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(small)).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1)[self._bgr2rgb].unsqueeze(0).float().div_(255.0)
        x = ((x - self._mean) / self._std).to(self.dtype)
        x = x.contiguous(memory_format=torch.channels_last)
        prob = self.net(x).float().softmax(dim=1)[0]
        p_out = float((prob * self._outdoor).sum())  # 여기서 GPU 동기화
        pk, ik = prob.topk(self.topk)
        top = [(self.names[i], p, bool(self._outdoor_np[i]))
               for p, i in zip(pk.tolist(), ik.tolist())]
        return SceneResult(p_out, top, (time.perf_counter() - t0) * 1000)

    def warmup(self, n: int = 10) -> float:
        """예열 후 평균 소요 시간(ms). 첫 호출은 cuDNN 알고리즘 선택으로 느리다."""
        dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        for _ in range(3):
            self.classify(dummy)
        ms = float(np.mean([self.classify(dummy).ms for _ in range(n)]))
        logger.info("장면 판별기 예열: 평균 %.1fms", ms)
        return ms


class SceneGate:
    """
    P(실외) 시계열 → 확정 장면. 문턱(현관, 출입문)에서 판정이 프레임마다 뒤집히면
    깊이 모델이 따라 바뀌면서 핀이 요동치므로, 세 장치를 겹쳐 둔다.

      1. 시간 EMA (tau_s)       순간 흔들림 제거. 호출 주기와 무관하게 초 단위로 동작
      2. 히스테리시스            실외로 가려면 p > enter_outdoor, 실내로 가려면 p < enter_indoor
      3. 체류 시간 (dwell_s)     2의 조건이 dwell_s 동안 끊기지 않아야 전환

    uncertain = 전환 대기 중이거나 p가 두 문턱 사이. 이때는 두 모델을 모두 돌려
    셀마다 더 가까운 값을 쓰라는 신호다(틀려도 위험한 쪽으로 틀리지 않게).
    """

    def __init__(self, tau_s: float = 1.0, enter_outdoor: float = 0.65,
                 enter_indoor: float = 0.35, dwell_s: float = 1.5,
                 initial: str | None = None) -> None:
        if not 0.0 < enter_indoor < enter_outdoor < 1.0:
            raise ValueError(f"문턱은 0 < enter_indoor({enter_indoor}) < "
                             f"enter_outdoor({enter_outdoor}) < 1 이어야 함")
        if tau_s < 0 or dwell_s < 0:
            raise ValueError("tau_s, dwell_s는 0 이상")
        if initial not in (None, "indoor", "outdoor"):
            raise ValueError(f"initial은 None/indoor/outdoor: {initial}")
        self.tau_s, self.dwell_s = tau_s, dwell_s
        self.enter_outdoor, self.enter_indoor = enter_outdoor, enter_indoor
        self.state: str | None = initial  # None = 첫 관측으로 결정
        self.p: float | None = None  # 평활된 P(실외)
        self.n_switches = 0
        self.last_switch_t: float | None = None
        self._t_last: float | None = None
        self._cand_since: float | None = None

    @property
    def other(self) -> str:
        return "outdoor" if self.state == "indoor" else "indoor"

    @property
    def pending(self) -> bool:
        """반대 장면으로의 전환 조건이 충족된 채 체류 시간을 세는 중."""
        return self._cand_since is not None

    def pending_progress(self, now: float) -> float:
        """체류 시간 진행률 0~1. 대기 중이 아니면 0."""
        if self._cand_since is None:
            return 0.0
        if self.dwell_s <= 0:
            return 1.0
        return min(1.0, (now - self._cand_since) / self.dwell_s)

    @property
    def uncertain(self) -> bool:
        if self.p is None or self.state is None:
            return True
        return self.pending or self.enter_indoor <= self.p <= self.enter_outdoor

    def observe(self, p_outdoor: float, now: float | None = None) -> str:
        """관측 1개를 반영하고 현재 확정 장면을 반환."""
        now = time.monotonic() if now is None else now
        p_outdoor = min(1.0, max(0.0, float(p_outdoor)))

        if self.p is None:
            self.p = p_outdoor
        elif self._t_last is not None:
            dt = max(0.0, now - self._t_last)
            a = 1.0 if self.tau_s == 0 else 1.0 - math.exp(-dt / self.tau_s)
            self.p += a * (p_outdoor - self.p)
        self._t_last = now

        if self.state is None:
            self.state = "outdoor" if self.p >= 0.5 else "indoor"
            return self.state

        want = (self.state == "indoor" and self.p > self.enter_outdoor) or (
            self.state == "outdoor" and self.p < self.enter_indoor)
        if not want:
            self._cand_since = None
            return self.state
        if self._cand_since is None:
            self._cand_since = now
        if now - self._cand_since >= self.dwell_s:
            self.state = self.other
            self._cand_since = None
            self.n_switches += 1
            self.last_switch_t = now
            logger.info("장면 전환 → %s (P(실외) 평활 %.2f)", self.state, self.p)
        return self.state

    def pause(self) -> None:
        """관측이 끊길 때(일시정지 등) 호출. 끊긴 시간이 EMA·체류 시간에 섞이지 않게."""
        self._t_last = None
        self._cand_since = None
