# 모델과 실행 자산 배치 가이드

이 저장소는 모델 가중치와 실행 바이너리를 포함하지 않습니다. 아래 디렉터리를
로컬에서 만들고 합법적으로 내려받은 자산을 배치한 뒤, `project-gpu/runtime.env`의
경로와 API 키를 맞추세요.

## 설정 파일 우선순위

각 실행 스크립트는 다음 순서로 설정을 읽습니다.

```text
.env.auto → project-gpu/runtime.env → 서비스별 .env
```

나중에 읽은 파일이 앞선 값을 덮어쓴다. `project-gpu/runtime.env.example`이 공개
공통 기준이며, 서비스별 `.env.example`은 꼭 필요한 차이만 넣는 선택적 override
예시입니다. 자동 생성 파일인 `.env.auto`를 직접 수정하기보다 정본을 복사한
`runtime.env`에 공통값을 두세요.

기본 호스트는 `127.0.0.1`로 유지합니다. 원격 연결을 명시적으로 구성하지 않는 한
`0.0.0.0`로 바꾸지 마세요. `EMBEDDING_API_KEY`는 임베딩 서버와 백엔드에서 같은
값이어야 하고, `LLM_API_KEY`도 llama-server와 백엔드에서 같은 값이어야 합니다.

## LLM: llama.cpp용 GGUF

포함된 LLM 기동 스크립트는 `llama.cpp`의 `llama-server`와 단일 `.gguf` 파일을
사용합니다. canonical 배치는 다음과 같습니다.

```text
project-gpu/main-backend/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf
project-gpu/main-backend/runtime/llama-server
```

관련 변수:

- `LLM_MODELS_DIR`: GGUF를 탐색할 상위 디렉터리
- `LLM_MODEL_PATH`: 비어 있지 않으면 자동 재귀 탐지보다 우선하는 정확한 GGUF 경로
- `LLM_MODEL_NAME`: OpenAI 호환 API에서 노출하고 백엔드가 요청할 model alias
- `LLM_RUNTIME`: 실행 가능한 Linux `llama-server` 경로
- `LLM_CTX_SIZE`: llama-server가 할당하는 context 크기
- `LLM_CONTEXT_LIMIT`: 백엔드가 허용하는 context 상한
- `LLM_API_KEY`: LLM 서버와 백엔드가 공유하는 인증 키

공개 예제의 기준은 `LLM_CTX_SIZE=131072`와 `LLM_CONTEXT_LIMIT=131072`입니다. 두
값을 같은 값으로 정렬하세요. GPU 메모리가 부족하면 두 값을 함께 낮추고 실제
모델의 지원 범위를 확인합니다.

`LLM_MODEL_PATH`가 비어 있을 때만 `LLM_MODELS_DIR` 아래 지원 패턴을 재귀 탐지해
GGUF를 선택합니다. 다른 Qwen GGUF나 llama.cpp 호환 GGUF도 쓸 수 있지만, 파일명과
`LLM_MODEL_NAME`을 함께 갱신하고 context 지원 범위를 확인해야 합니다. PyTorch
`.safetensors` 체크포인트는 이 llama.cpp 실행 경로의 입력이 아닙니다.

Linux CUDA runtime은 다음 스크립트로 설치하거나 빌드할 수 있습니다.

```bash
bash project-gpu/install_llama_cuda_runtime.sh --mode release --cuda 12.4
bash project-gpu/install_llama_cuda_runtime.sh --mode build --cuda-arch 90
bash project-gpu/build_llama_runtime_offline.sh --cuda on --cuda-arch 90 --keep-build
```

Windows용 `llama-server.exe`는 Linux GPU 서버의 대체물이 아닙니다. GPU와 드라이버,
CUDA 버전에 맞는 native Linux 바이너리와 공유 라이브러리를 준비하세요.

## 임베딩: Hugging Face/safetensors 디렉터리

canonical 임베딩 디렉터리는 다음과 같습니다.

```text
project-gpu/embedding-gpu-server/models/Qwen/Qwen3-Embedding-0.6B/
```

폴더 안에는 Hugging Face에서 로드 가능한 `config.json`, tokenizer 설정/어휘 파일,
그리고 하나 이상의 `.safetensors` 가중치와 필요한 메타데이터가 있어야 합니다.
이 safetensors/Hugging Face 디렉터리는 임베딩 서버용이며 llama.cpp LLM 입력이
아니다.

관련 이름과 실제 환경 변수:

- `EMBEDDING_MODEL_LARGE_PATH`: 위 모델 폴더의 정확한 경로
- `DEFAULT_INDEX` (`EMBEDDING_DEFAULT_INDEX`): 요청에 index가 없을 때의 기본 모델
- `API_LARGE_ALIAS` (`EMBEDDING_API_LARGE_ALIAS`): large 모델의 API alias
- `MODEL_DEVICE` (`EMBEDDING_MODEL_DEVICE`): `cuda`, `cpu` 등 모델 로드 장치
- `EMBED_BATCH_SIZE`: 임베딩 모델 내부 배치 크기
- `EMBEDDING_API_URL`: 백엔드가 호출할 임베딩 서버 주소
- `API_BATCH_SIZE` (`EMBEDDING_API_BATCH_SIZE`): 백엔드의 API 요청 배치 크기
- `API_KEY` (`EMBEDDING_API_KEY`): 임베딩 서버와 백엔드가 공유할 인증 키

`Qwen__Qwen3-Embedding-0.6B`, `kure-v1`, `multilingual-e5-large`,
`Qwen3-Embedding-4B`, `jina-embeddings-v5-text-small` 등의 기존 탐지 패턴도
지원됩니다. 대체 모델을 쓰면 임베딩 차원과 의존성 요구사항을 확인하고 기존
인덱스를 새 모델로 다시 생성하세요.

## OCR: PaddleOCR-VL 로컬 디렉터리

기본 OCR 구성은 네 디렉터리를 사용합니다.

```text
project-gpu/main-backend/models/ocr/PaddleOCR-VL
project-gpu/main-backend/models/ocr/PP-DocLayoutV3
project-gpu/main-backend/models/ocr/PP-LCNet_x1_0_doc_ori
project-gpu/main-backend/models/ocr/UVDoc
```

변수 매핑:

- `PDF_OCR_MODEL_NAME`, `PDF_OCR_VL_MODEL_DIR` → `PaddleOCR-VL`
- `PDF_OCR_LAYOUT_MODEL_DIR` → `PP-DocLayoutV3`
- `PDF_OCR_DOC_ORIENTATION_MODEL_DIR` → `PP-LCNet_x1_0_doc_ori`
- `PDF_OCR_DOC_UNWARP_MODEL_DIR` → `UVDoc`
- `PDF_OCR_BACKEND`: 사용할 OCR backend
- `PDF_OCR_DEVICE`: `gpu`, `cuda...`, `cpu` 등 실행 장치
- `PDF_OCR_ALLOW_ONLINE_MODEL_FALLBACK=0`: 누락 모델을 실행 중 인터넷에서 받지 않음

경로가 모델 폴더 자체를 가리키는지 확인하세요. 상위 `models/ocr`만 지정하면 각
구성 요소를 찾지 못할 수 있습니다. 다른 호환 PaddleOCR 모델을 쓸 때에는 네 역할과
각 환경 변수를 개별적으로 맞춰야 합니다.

## 배치 확인

설정 로드 결과와 파일을 먼저 확인할 수 있습니다.

```bash
bash project-gpu/load_gpu_env.sh --print
test -f project-gpu/main-backend/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf
test -x project-gpu/main-backend/runtime/llama-server
test -f project-gpu/embedding-gpu-server/models/Qwen/Qwen3-Embedding-0.6B/config.json
find project-gpu/embedding-gpu-server/models/Qwen/Qwen3-Embedding-0.6B \
  -maxdepth 2 -name '*.safetensors' -print
test -d project-gpu/main-backend/models/ocr/PaddleOCR-VL
test -d project-gpu/main-backend/models/ocr/PP-DocLayoutV3
test -d project-gpu/main-backend/models/ocr/PP-LCNet_x1_0_doc_ori
test -d project-gpu/main-backend/models/ocr/UVDoc
bash project-gpu/check_gpu_assets.sh
```

`check_gpu_assets.sh`는 정보성 검사기입니다. `[MISS]`가 하나라도 출력되면 필수
자산 누락으로 간주하고 시작을 중단하세요. `[MISS]`가 있어도 현재 checker의 exit
code가 0일 수 있으므로 종료 코드만으로 성공을 판단하지 않습니다.

## Git에 넣지 않는 이유

GGUF, safetensors, tokenizer, OCR 모델, `llama-server`와 그 라이브러리는 용량이
크고 각자 별도 라이선스와 배포 조건이 있습니다. 모델과 런타임은 이 문서의 로컬
경로에 두되 커밋하지 마세요. `models/`, `runtime/`, `offline_assets/`,
`offline_packages/`는 공개 manifest에서도 제외됩니다.
