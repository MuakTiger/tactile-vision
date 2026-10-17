"""
Depth Anything V2 metric TensorRT 추론 + GPU 셀 풀링.

기존 코드 대비 바뀐 점
─────────────────────
1. 전처리를 GPU에서 한다. 기존은 CPU numpy로 BGR→RGB, /255, 정규화, transpose를
   매 프레임 수행했다(504×280×3 float32 = 1.7MB 연산 × 4단계).
2. 추론 결과 전체를 CPU로 내리지 않는다. 기존은 518×518 float32 = 1.07MB를
   매 프레임 D2H 복사한 뒤 CPU에서 평균을 냈다. 여기서는 GPU에서 24개 셀 대표값을
   뽑아 96바이트만 내린다.
3. 셀 대표값을 평균이 아니라 하위 분위수로 뽑는다. 평균을 쓰면 기둥·의자다리 같은
   얇은 장애물이 배경에 묻혀 사라진다. 장애물 회피에 필요한 값은 '셀 안에서
   가장 가까운 표면'이다.
4. execute_async_v3 후 스트림 동기화를 반드시 한다. 기존 main.py/main4.py는
   동기화 없이 .cpu()를 호출해 추론 미완료 버퍼를 읽을 수 있었다.
5. (2026-10-17) 출력 → 미터 변환 두 가지를 셀 풀링 전에 넣을 수 있다.
   offset_m: metric 출력에 더함(실내 모델 −0.44 m 보정).
   ground:   상대깊이 출력 r을 땅 기준 배율 A로 A / r 미터로 바꿈(ground.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from . import geometry as geo
from .config import DepthCfg, GridCfg
from .ground import GroundScaler

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthEngine:
    """
    TRT 엔진 래퍼. 입력 BGR uint8 (H,W,3) → 셀 거리 (24,) 미터.

    전체 깊이맵이 필요하면 last_depth_map()으로 가져온다(디버그용, 비용 발생).
    """

    def __init__(self, dcfg: DepthCfg, gcfg: GridCfg, engine_path: Path,
                 offset_m: float = 0.0, ground: GroundScaler | None = None) -> None:
        self.dcfg = dcfg
        self.gcfg = gcfg
        self.spec = dcfg.spec
        # 출력 → 미터: ground가 있으면 상대깊이(A / r), 없으면 metric 출력 + offset_m
        self.offset_m = float(offset_m)
        self.ground = ground
        if ground is not None and self.offset_m:
            raise ValueError("ground 환산과 offset_m을 함께 쓸 수 없음")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없음")
        if not engine_path.exists():
            raise FileNotFoundError(
                f"엔진 없음: {engine_path}\n"
                f"생성: python tools/build_engine.py --spec {dcfg.input_spec}"
            )

        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"엔진 역직렬화 실패: {engine_path}. "
                "TensorRT 버전이 빌드 시점과 다를 수 있음 — 재빌드 필요."
            )
        self.context = self.engine.create_execution_context()

        # 엔진 실제 shape과 설정이 일치하는지 검증 (기존 코드엔 이 검증이 없었다)
        self._bind_names()
        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        want = (1, 3, self.spec.height, self.spec.width)
        if in_shape != want:
            raise RuntimeError(
                f"엔진 입력 shape {in_shape} != 설정 {want}. "
                f"config의 depth.input_spec('{dcfg.input_spec}')과 엔진이 불일치."
            )
        out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        logger.info("엔진 로드: %s  입력%s 출력%s", engine_path.name, in_shape, out_shape)

        self.stream = torch.cuda.Stream()
        dev = "cuda"
        self.d_input = torch.empty(want, dtype=torch.float32, device=dev)
        # 출력은 (1,H,W) 또는 (1,1,H,W) 둘 다 올 수 있으므로 실제 shape을 따른다
        self.d_output = torch.empty(out_shape, dtype=torch.float32, device=dev)
        self.context.set_tensor_address(self.in_name, int(self.d_input.data_ptr()))
        self.context.set_tensor_address(self.out_name, int(self.d_output.data_ptr()))

        self._depth_buf = torch.empty((self.spec.height, self.spec.width), dtype=torch.float32, device=dev)
        self._mean = torch.tensor(IMAGENET_MEAN, device=dev).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=dev).view(3, 1, 1)
        # BGR→RGB 채널 재배열 인덱스
        self._bgr2rgb = torch.tensor([2, 1, 0], device=dev)

        self._cell_slices = self._build_cell_slices()
        self._k = self._percentile_k()
        logger.info(
            "셀 풀링: %dx%d px/셀, %s %.0f%% → k=%d",
            self.spec.cell_w, self.spec.cell_h, gcfg.reducer, gcfg.percentile, self._k,
        )

    def _bind_names(self) -> None:
        ins, outs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                ins.append(n)
            else:
                outs.append(n)
        if len(ins) != 1 or len(outs) != 1:
            raise RuntimeError(f"입출력 텐서가 1개씩이 아님: in={ins} out={outs}")
        self.in_name, self.out_name = ins[0], outs[0]

    # ── 셀 분할 ──
    def _build_cell_slices(self) -> tuple[slice, slice]:
        """cell_margin을 반영한 셀 내부 영역 슬라이스."""
        m = self.gcfg.cell_margin
        ch, cw = self.spec.cell_h, self.spec.cell_w
        dy, dx = int(ch * m), int(cw * m)
        return slice(dy, ch - dy if dy else ch), slice(dx, cw - dx if dx else cw)

    def _cell_pixels(self) -> int:
        sy, sx = self._cell_slices
        return len(range(*sy.indices(self.spec.cell_h))) * len(
            range(*sx.indices(self.spec.cell_w))
        )

    def _percentile_k(self) -> int:
        """분위수를 kthvalue의 k로 변환. quantile보다 빠르고 정확하다."""
        n = self._cell_pixels()
        k = int(round(n * self.gcfg.percentile / 100.0))
        return max(1, min(n, k))

    # ── 전처리 (GPU) ──
    def _preprocess(self, bgr: np.ndarray) -> None:
        if bgr.shape[:2] != (self.spec.height, self.spec.width):
            raise ValueError(
                f"입력 해상도 불일치: {bgr.shape[:2]} != "
                f"{(self.spec.height, self.spec.width)}"
            )
        # uint8 HWC 업로드 (423KB) — float32 CHW 업로드(1.7MB)보다 4배 적다
        t = torch.from_numpy(np.ascontiguousarray(bgr)).to(
            "cuda", non_blocking=True
        )
        t = t.permute(2, 0, 1)[self._bgr2rgb].float().div_(255.0)
        t.sub_(self._mean).div_(self._std)
        self.d_input[0].copy_(t)

    # ── 셀 풀링 (GPU) ──
    def _pool_cells(self, depth: torch.Tensor) -> torch.Tensor:
        """
        depth: (H, W) 미터
        반환:  (24,) 셀 대표 거리. 행 우선(좌→우, 위→아래).
        """
        R, C = geo.GRID_ROWS, geo.GRID_COLS
        ch, cw = self.spec.cell_h, self.spec.cell_w
        sy, sx = self._cell_slices
        # (R, ch, C, cw) → (R, C, ch, cw) → (24, ch*cw)
        blocks = depth.view(R, ch, C, cw).permute(0, 2, 1, 3)
        blocks = blocks[:, :, sy, sx].reshape(R * C, -1)

        if self.gcfg.reducer == "mean":
            return blocks.mean(dim=1)
        if self.gcfg.reducer == "min":
            return blocks.min(dim=1).values
        # percentile: k번째 최솟값 = 가까운 쪽 분위수
        return blocks.kthvalue(self._k, dim=1).values

    # ── 출력 → 미터 ──
    def _to_metres(self, out: torch.Tensor, now: float | None) -> torch.Tensor:
        """엔진 출력 (H, W) → 미터 깊이맵. 같은 스트림 안에서 부른다."""
        if self.ground is not None:
            a = self.ground.update(out, now)  # .item()이 이 스트림의 추론 완료를 기다린다
            buf = torch.clamp(out, min=1e-6, out=self._depth_buf)  # r ≤ 0(하늘) → 아주 먼 거리
            return buf.reciprocal_().mul_(a)  # a가 NaN(처음부터 무효)이면 전부 NaN = 무효 셀
        if self.offset_m:
            # 보정 뒤 0 이하가 되면(아주 가까움) 1 cm로 — 근거리 최대 단계로 간다
            return torch.add(out, self.offset_m, out=self._depth_buf).clamp_(min=0.01)
        return out

    def reset(self) -> None:
        """시간 상태(땅 기준 배율 평활) 초기화. 예열 뒤에 부른다."""
        if self.ground is not None:
            self.ground.reset()

    # ── 추론 ──
    @torch.inference_mode()
    def infer(self, bgr: np.ndarray, now: float | None = None) -> np.ndarray:
        """BGR (H,W,3) uint8 → 셀 거리 (24,) float32 미터. now: 프레임 시각(땅 기준 평활용)."""
        with torch.cuda.stream(self.stream):
            self._preprocess(bgr)
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TRT 추론 실행 실패")
            out = self.d_output.view(self.spec.height, self.spec.width)
            depth = self._to_metres(out, now)
            cells = self._pool_cells(depth)
        # 반드시 동기화 후에 읽는다 — 이 한 줄이 없으면 미완료 버퍼를 읽는다
        self.stream.synchronize()
        self._last_depth = depth
        return cells.cpu().numpy().astype(np.float32)

    def last_depth_map(self) -> np.ndarray:
        """직전 프레임의 전체 깊이맵 (미터). 디버그 시각화용 — D2H 비용 발생."""
        return self._last_depth.cpu().numpy()

    # ── 진단 ──
    def warmup(self, n: int = 10) -> float:
        """엔진 예열 후 평균 추론시간(ms) 반환. 첫 수 회는 느리므로 반드시 호출."""
        import time

        dummy = np.zeros((self.spec.height, self.spec.width, 3), dtype=np.uint8)
        for _ in range(3):
            self.infer(dummy)
        t0 = time.perf_counter()
        for _ in range(n):
            self.infer(dummy)
        ms = (time.perf_counter() - t0) / n * 1000
        self.reset()  # 빈 화면으로 잡힌 땅 기준 배율을 버린다
        logger.info("예열 완료: 평균 추론 %.1fms (%.1f fps 상당)", ms, 1000 / ms)
        return ms
