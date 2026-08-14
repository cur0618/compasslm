# project-gpu 운영 가이드 (GPU-only, 2026-02-16)

`project-gpu`는 임베딩 서버(FastAPI)와 백엔드/LLM 서버를 분리해 운영하는 단일 트랙이다.
로컬(`project-local`) 트랙은 폐기되었고 본 문서 기준으로만 운영한다.

공개 저장소의 공통 기준 설정은 `runtime.env.example`이다. 먼저
`cp project-gpu/runtime.env.example project-gpu/runtime.env`로 복사하고 실제 경로와
비밀 키를 입력한다. 서비스별 `.env.example`은 필요한 값만 나중에 덮어쓸 때
사용한다. 모델 형식과 정확한 배치 경로는
[`../docs/MODELS_AND_ASSETS.md`](../docs/MODELS_AND_ASSETS.md)를 참고한다.

## 1) 경로 준비
```bash
cd ~/compasslm
chmod +x project-gpu/*.sh project-gpu/embedding-gpu-server/*.sh
project-gpu/load_gpu_env.sh --print
```

핵심 경로:
- 임베딩 서버: `project-gpu/embedding-gpu-server`
- 메인 백엔드: `project-gpu/main-backend`
- 공통 백엔드 코드: `src`

## 2) LLM 모델 탐지 규칙
`project-gpu/load_gpu_env.sh`의 `compass_detect_llm_model_path()`는 아래 순서로 자동 탐지한다.
1. `qwen*3.5*9b*.gguf`
2. `qwen*3.5*9b*q4*k*m*.gguf`
3. `qwen*3.5*9b*instruct*q4*k*m*.gguf`
4. `qwen*3*9b*q4*k*m*.gguf`
5. `qwen*3*9b*instruct*q4*k*m*.gguf`
6. `qwen*3*32b*q4*k*m*.gguf`
7. `qwen*3*32b*instruct*q4*k*m*.gguf`
8. `qwen*2.5*14b*instruct*q4*k*m*.gguf`
9. `qwen*14b*instruct*q4*k*m*.gguf`
10. `gemma*3n*e4b*q4*k*m*.gguf`
11. `models/llm` 하위(최대 4-depth)의 첫 번째 `*.gguf`
12. 고정 경로가 없으면 재귀 탐지 fallback (최대 5-depth)

탐지 실패 시 fallback:
- `project-gpu/main-backend/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf`

명시 고정이 필요하면:
- `project-gpu/main-backend/.env`에 `LLM_MODEL_PATH=/absolute/path/to/model.gguf`

## 3) 필수 자산 배치
- 임베딩 모델:
  - 위치: `project-gpu/embedding-gpu-server/models`
  - 지원 예시(모델별 하위 폴더 분기):
    - `project-gpu/embedding-gpu-server/models/Qwen/Qwen3-Embedding-0.6B/...`
    - `project-gpu/embedding-gpu-server/models/Qwen__Qwen3-Embedding-0.6B/...`
    - `project-gpu/embedding-gpu-server/models/kure-v1/...`
    - `project-gpu/embedding-gpu-server/models/nlpai-lab/kure-v1/...`
    - `project-gpu/embedding-gpu-server/models/multilingual-e5-large/...`
    - `project-gpu/embedding-gpu-server/models/qwen3-embedding-4b/...`
    - `project-gpu/embedding-gpu-server/models/Qwen/Qwen3-Embedding-4B/...`
    - `project-gpu/embedding-gpu-server/models/jinaai/jina-embeddings-v5-text-small/...`
  - 자동 선택 우선순위: `qwen3-embedding-0.6b` -> `kure-v1` 계열 -> `multilingual-e5-large` -> `qwen3-embedding-4b` -> `jina-embeddings-v5-text-small` -> 기타 첫 모델 폴더
  - 내부 폴더를 추가로 내려가며 탐지(최대 4-depth)
  - 고정 경로가 비어 있으면 상위 폴더 기준으로 다시 탐지 fallback
  - `Qwen/Qwen3-Embedding-0.6B`는 공식 카드 기준 `transformers>=4.51.0`, `sentence-transformers>=2.7.0`이면 충분하고, 현재 오프라인 번들 검증도 이 기준으로 맞춤
  - `jinaai/jina-embeddings-v5-text-small`를 쓰려면 임베딩 venv에 `transformers>=4.57.0`, `torch>=2.8.0`, `peft>=0.15.2`가 필요하므로 번들 재생성이 안전함
- LLM GGUF:
  - 위치: `project-gpu/main-backend/models/llm`
  - 지원 예시(모델별 하위 폴더 분기):
    - `project-gpu/main-backend/models/llm/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf`
    - `project-gpu/main-backend/models/llm/qwen3.0-32b/qwen3.0-32b-q4_k_m.gguf`
    - `project-gpu/main-backend/models/llm/qwen2.5-14b/qwen2.5-14b-instruct-q4_k_m.gguf`
  - 내부 폴더를 추가로 내려가며 탐지(최대 5-depth)
- llama-server 런타임:
  - 우선: `project-gpu/main-backend/runtime/llama-server`
  - 대체: `runtime/llama-server`
  - 참고: `llama-server.exe`는 Windows/WSL 전용이며, Linux GPU 서버에서는 native Linux 바이너리가 필요

### CUDA runtime 교체 스크립트
```bash
cd ~/compasslm
project-gpu/install_llama_cuda_runtime.sh
```

옵션 예시:
```bash
# 릴리스 CUDA 자산만 사용(없으면 실패)
project-gpu/install_llama_cuda_runtime.sh --mode release --cuda 12.4

# 소스에서 직접 CUDA 빌드
project-gpu/install_llama_cuda_runtime.sh --mode build --jobs 16 --cuda-arch 90
```

H100 서버에서는 `--cuda-arch 90`이 기본값입니다. 기존 런타임에서
`no kernel image is available for execution on the device`가 발생하면 릴리스
바이너리 대신 위 source build 경로로 `llama-server`를 다시 빌드하세요.

### GCC 8 오프라인 빌드 참고 (`std::filesystem` 링크 오류 대응)
- `libstdc++fs` 파일을 미리 배치:
  - 권장: `project-gpu/offline_assets/toolchain/libstdc++fs.a`
- 오프라인 빌드 시 자동 탐지되며, 필요하면 직접 지정:
```bash
cd ~/compasslm
project-gpu/build_llama_runtime_offline.sh --cuda on --cuda-arch 90 --keep-build \
  --stdcxxfs-lib ~/compasslm/project-gpu/offline_assets/toolchain/libstdc++fs.a
```

## 4) Python 환경 설치
온라인 설치:
```bash
cd ~/compasslm
project-gpu/setup_gpu_track.sh --python python3.11
```

GPU PaddlePaddle wheel은 CPython 3.11용이므로 온라인 GPU 설치도 Python 3.11을
사용한다. `runtime.env`의 `BACKEND_PADDLE_RUNTIME_KIND`와
`BACKEND_PADDLE_CUDA_TRACK`을 설치 전에 확인한다.

임베딩 서버만 오프라인 설치(cp311):
```bash
cd ~/compasslm
project-gpu/setup_gpu_track.sh --python python3.11 --offline-embed
```

임베딩+백엔드 모두 오프라인 설치(cp311, 동일 wheel 폴더 사용):
```bash
cd ~/compasslm
project-gpu/setup_gpu_track.sh --python python3.11 --offline-embed --offline-backend
```

생성/설치 대상:
- `project-gpu/embedding-gpu-server/compassvenv`
- `project-gpu/main-backend/compassvenv`
- `project-gpu/embedding-gpu-server/.env.auto`
- `project-gpu/main-backend/.env.auto`
- 백엔드 의존성 기준 파일: `project-gpu/main-backend/requirements.txt`

백엔드 메모:
- 메인 채팅 오케스트레이션은 `PydanticAI`를 사용한다.
- OpenAI 호환 LLM 서버를 쓸 때는 `.env`에 `LLM_MODEL_NAME`을 실제 서버 model alias와 맞춰 두는 것을 권장한다.
- 브라우저 세션 대화 이력은 기본적으로 `data/app.sqlite`에 저장된다.
- PDF는 기본적으로 `PDF_PARSE_MODE=ocr_first`로 `PaddleOCR-VL` 결과를 먼저 색인한다. `PyMuPDF`는 OCR 실패, 빈 OCR 결과, OCR page limit 초과분의 보조 fallback으로만 사용한다.
- V100 32GB 기준 fast profile은 `PDF_OCR_OPTIMIZATION_PROFILE=v100_32gb_fast`이며, 기본 목표는 `PDF_OCR_TARGET_PAGES=200`, `PDF_OCR_TARGET_SECONDS=300`이다.
- `PDF_OCR_GPU_PROCESS_ISOLATION=1`은 GPU OCR을 별도 worker process에서 실행해 OCR 완료 후 Paddle/CUDA VRAM을 process 종료로 회수하기 위한 기본값이다.
- 웹 업로드는 업로드마다 별도 `.sh`를 실행하지 않는다. `run_backend_api.sh`가 OCR env를 읽고 FastAPI를 기동한 뒤, `/upload` 요청은 백엔드 Python 프로세스 안에서 `src.main` 업로드 큐 -> `RAGEngine.ingest_file()` -> `src.pdf_ocr.extract_pdf_pages()` 경로로 처리된다. `benchmark_pdf_ocr_v100.sh`와 `tune_pdf_ocr_v100_matrix.sh`는 같은 OCR 경로를 직접 호출하는 실측/튜닝 도구다.
- 오프라인 번들에는 `PaddleOCR`, `PaddlePaddle GPU runtime`, `PyMuPDF` wheel과 로컬 OCR 모델 디렉터리가 함께 있어야 한다.

## 5) 자산 점검
```bash
cd ~/compasslm
project-gpu/check_gpu_assets.sh
```

`MISS`가 있으면 실행 전에 해당 자산을 채운다.

V100 32GB PDF OCR 실측 전 사전 점검:
```bash
cd ~/compasslm
project-gpu/preflight_pdf_ocr_v100.sh
```

200페이지/5분 OCR 목표 실측과 튜닝:
```bash
cd ~/compasslm
project-gpu/run_pdf_ocr_v100_acceptance.sh /path/to/200-page.pdf
project-gpu/verify_pdf_ocr_v100_acceptance_report.sh logs/ocr-benchmarks/<run>_v100_acceptance/acceptance_report.json
project-gpu/tune_pdf_ocr_v100_matrix.sh /path/to/200-page.pdf
project-gpu/verify_pdf_ocr_v100_target.sh logs/ocr-benchmarks/<run>_v100_matrix/summary.json
project-gpu/apply_pdf_ocr_tuned_profile.sh logs/ocr-benchmarks/<run>_v100_matrix/summary.json
project-gpu/apply_pdf_ocr_tuned_profile.sh logs/ocr-benchmarks/<run>_v100_matrix/summary.json project-gpu/runtime.env --write
```

`summary.json`의 `target_achieved=true`가 실제 목표 달성 기준이다. 더 엄격하게 확인하려면 `verify_pdf_ocr_v100_target.sh summary.json [max-peak-gpu-mb] [max-process-rss-mb]`를 실행해 `ocr_pages>=200`, `elapsed_seconds<=300`, 선택적 peak VRAM/process tree RSS 상한을 함께 검증한다. `run_pdf_ocr_v100_acceptance.sh` 통과 시 `logs/ocr-benchmarks/<run>_v100_acceptance/acceptance_report.json`이 최종 증거 파일이며, checklist의 `ocr_pages_ge_target`, `elapsed_seconds_le_target`, `peak_gpu_le_limit`, `process_rss_le_limit`를 확인한다.

## 6) 서버 실행 순서
실행 전에 먼저 아래를 확인:
```bash
cd ~/compasslm
project-gpu/check_gpu_assets.sh
project-gpu/preflight_pdf_ocr_v100.sh
```

터미널 1: 임베딩 API
```bash
cd ~/compasslm
project-gpu/run_embedding_server.sh
```
확인: 모델 로드와 인증 포함 `/embed` probe가 끝나면 `[READY] embedding url=http://127.0.0.1:<선택포트>`가 출력됩니다. 이 로그 전에 backend를 시작하면 backend는 즉시 실패합니다.

터미널 2: LLM 서버
```bash
cd ~/compasslm
# Long-context validation example:
# LLM_CTX_SIZE=131072 LLM_CONTEXT_LIMIT=131072 project-gpu/run_llm_server.sh
project-gpu/run_llm_server.sh
```
확인: `[READY] llm url=http://127.0.0.1:<선택포트>/v1/chat/completions` 로그가 출력됩니다.

참고: LLM 루트 주소는 브라우저용 화면이 아닐 수 있어 `endpoint not found`나 404가 나와도 정상일 수 있습니다. backend는 ready state에 기록된 실제 `/v1/chat/completions` URL만 사용합니다.

터미널 3: 백엔드 API
```bash
cd ~/compasslm
API_RELOAD=1 project-gpu/run_backend_api.sh
```

확인: `[READY] backend url=http://127.0.0.1:<선택포트>` 로그가 출력됩니다.

자동 포트 모드에서 backend는 runtime state의 준비 완료 embedding URL만 사용합니다. 준비 완료 state가 없거나 PID, `/health`, 인증, `/embed` probe가 실패하면 `.env`의 `8002`로 fallback하지 않고 기동을 중단합니다.

## 7) 원격 GPU 임베딩 서버 사용
기본값은 같은 컴퓨터의 `127.0.0.1` 연결이다. 임베딩 서버를 원격에서 제공할 때만
아래처럼 노출 범위를 의도적으로 바꾼다. 서비스별 .env 파일은 `.env.auto`와
`runtime.env`보다 마지막에 읽혀 앞선 공통값을 덮어쓴다.

임베딩 서버의 최종 `project-gpu/embedding-gpu-server/.env`:

```bash
COMPASS_AUTO_PORT=0
EMBED_HOST=0.0.0.0
EMBED_PORT=8002
EMBEDDING_API_KEY=replace-with-strong-secret
```

백엔드의 최종 `project-gpu/main-backend/.env`:

```bash
COMPASS_AUTO_PORT=0
EMBEDDING_API_URL=https://embedding.example.com
EMBEDDING_API_KEY=replace-with-strong-secret
```

두 `EMBEDDING_API_KEY`는 같은 강한 값이어야 한다. `EMBED_HOST=0.0.0.0`은 모든
인터페이스에 bind하므로 firewall allowlist로 백엔드 호스트만 허용한다. plaintext
Bearer 인증을 public Internet에 직접 노출하지 않는다. TLS reverse proxy를 앞에
두고 `https://` URL을 사용하거나, VPN 또는 private tunnel 안에서만 연결한다.
`COMPASS_AUTO_PORT=0`인 백엔드는 기동 전에 원격 health/embed 검증을 통과해야 한다.

JupyterHub 등의 사용자별 reverse proxy를 통할 때 경로 예시는
`/user/<user>/proxy/8004/`처럼 일반화하고, 로컬 사용자명을 문서나 설정에
커밋하지 않는다.

## 8) 스모크 테스트
1. backend `[READY]` 로그에 출력된 URL로 접속
2. `.txt`/`.xlsx` 업로드
3. 질문 입력 후 근거 포함 응답 확인
