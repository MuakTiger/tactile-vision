# vendor/depth_anything_v2

Depth Anything V2 **metric depth** 모델 정의. 출처:
`Depth-Anything-V2/metric_depth/depth_anything_v2/` (Apache-2.0, 원본 라이선스 전문: `depth_anything_v2/LICENSE`)
https://github.com/DepthAnything/Depth-Anything-V2 — dinov2 부분은 Meta Platforms (Apache-2.0, 파일 머리말 유지)

여기에 복사해 둔 이유: 외부 디렉토리를 import하면 그 트리에 `__pycache__`가
생성되어 남의 작업 폴더를 오염시킨다. 이 프로젝트는 외부에 어떤 파일도 쓰지 않는다.

상대깊이용(`Depth-Anything-V2/depth_anything_v2/`)과 forward가 다르다.
metric 쪽은 sigmoid 출력에 `max_depth`를 곱해 미터를 직접 낸다.
