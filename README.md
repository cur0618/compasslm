# CompassLM

CompassLM은 로컬 모델로 문서를 색인하고, 검색 근거와 인용을 포함한 답변을 만드는
한국어 중심 RAG 애플리케이션입니다. 이 저장소는 다른 컴퓨터에서 구조와 설정을
재현할 수 있도록 소스, 테스트, 실행 스크립트, 안전한 예제 설정만 공개합니다.
모델이나 운영 데이터까지 포함한 완제품 배포본은 아닙니다.

## 구성과 데이터 흐름

백엔드/UI가 요청을 받아 임베딩과 LLM을 각각 호출하는 orchestrator입니다.

```text
사용자/업로드 → backend/UI orchestrator
backend → embedding API → 색인·질의 벡터
local data/SQLite/index → retrieval → 검색 근거 → backend
backend → llama.cpp LLM → 답변 생성 → backend
backend → 인용(citations) 포함 응답 → 사용자/UI
```

- 백엔드/UI는 업로드와 질문을 받고 전체 색인·검색·답변 흐름을 조정합니다.
- 백엔드는 문서 색인과 질의 검색을 위해 임베딩 API를 호출합니다.
- 로컬 SQLite/index가 retrieval에 검색 후보를 제공하고, 검색 결과가 답변 근거가
  됩니다.
- 백엔드는 그 근거로 llama.cpp LLM에 답변 생성을 요청한 뒤 인용을 붙여 UI에
  반환합니다.
- 업로드와 실행 중 생성되는 SQLite/index는 로컬 `data/`에만 남고 Git에 들어가지
  않습니다.

주요 소스 트리는 다음과 같습니다.

```text
src/                              백엔드, RAG, 웹 UI 공통 코드
tests/                            동작 및 공개 릴리스 계약 테스트
scripts/                          평가, 스모크 테스트, 릴리스 검사 도구
project-gpu/                      GPU 설치·기동·진단 스크립트
  embedding-gpu-server/           임베딩 FastAPI 서비스
  main-backend/                   백엔드 환경 및 LLM/OCR 자산 기준 경로
docs/MODELS_AND_ASSETS.md         로컬 모델 배치와 설정 기준
PUBLIC_RELEASE_MANIFEST.txt       공개 파일의 유일한 허용 목록
```

## 준비 사항과 온라인 설치

- Python 3.11+
- Bash를 사용할 수 있는 Linux
- GPU 경로는 NVIDIA GPU, 호환 드라이버와 CUDA 필요
- 저장 공간은 모델과 로컬 인덱스 크기를 별도로 고려

공통 설정을 먼저 복사해 PaddlePaddle runtime 종류와 CUDA track을 선택한 뒤,
서비스 실행에 사용하는 두 가상환경을 설치 스크립트로 준비합니다.

```bash
cp project-gpu/runtime.env.example project-gpu/runtime.env
# runtime.env의 BACKEND_PADDLE_RUNTIME_KIND와 BACKEND_PADDLE_CUDA_TRACK 확인
bash project-gpu/setup_gpu_track.sh --python python3.11
```

이 명령은 서비스 실행 스크립트가 실제로 사용하는 아래 환경을 만들고, 임베딩과
백엔드 요구사항 및 선택된 PaddlePaddle runtime을 설치합니다.

```text
project-gpu/embedding-gpu-server/compassvenv
project-gpu/main-backend/compassvenv
```

저장소 루트의 `.venv`는 전체 테스트나 개발 도구를 위한 선택 사항이며 서비스
실행용이 아닙니다. 필요할 때만 별도로 만드세요.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU별 설치, CUDA 런타임 빌드, 오프라인 패키지 준비는
[`project-gpu/README.md`](project-gpu/README.md)를 참고하세요.

## 설정과 모델 배치

위 설치 단계에서 복사한 실제 `runtime.env`는 Git에서 제외됩니다.
`project-gpu/runtime.env`에서 `replace-with-strong-secret` 키를 강한 무작위 값으로
바꾸고, LLM·임베딩·OCR 모델 경로와 장치 설정을 현재 컴퓨터에 맞춥니다. 서버와
그 서버를 호출하는 클라이언트의 API 키는 반드시 같은 값이어야 합니다.

설정은 `.env.auto → project-gpu/runtime.env → 서비스별 .env` 순서로 읽으며,
나중에 읽은 파일이 앞선 값을 덮어씁니다. 각 서비스의 `.env.example`은 선택적인
나중 덮어쓰기 예시입니다. 모든 예제를 무작정 `.env`로 복사하지 말고, 꼭 필요한
서비스별 차이만 추가하세요.

모델 파일 형식, 정확한 배치 경로, 변수별 의미는
[`docs/MODELS_AND_ASSETS.md`](docs/MODELS_AND_ASSETS.md)에 정리되어 있습니다.

## 자산 확인과 실행

먼저 자산 검사를 실행합니다.

```bash
bash project-gpu/check_gpu_assets.sh
```

출력에 `[MISS]`가 하나라도 있으면 해당 필수 자산을 채우기 전에는 시작하지
마세요. 이 검사기는 정보를 모아 보여 주는 도구라 `[MISS]`가 있어도 exit code 0을
반환할 수 있습니다. 따라서 종료 코드만 보지 말고 출력 자체를 확인해야 합니다.

서로 다른 세 터미널에서 아래 순서로 실행합니다.

```bash
# 터미널 1
bash project-gpu/run_embedding_server.sh
# [READY] embedding 확인
```

```bash
# 터미널 2
bash project-gpu/run_llm_server.sh
# [READY] llm 확인
```

```bash
# 터미널 3
bash project-gpu/run_backend_api.sh
# [READY] backend 확인
```

자동 포트 선택이 켜져 있으면 기본 포트와 달라질 수 있습니다. 각 `[READY]` 로그와
runtime state에 기록된 URL이 실제 주소의 기준입니다.

## 기본 상태 확인

기본 포트를 사용할 때 다음 probe로 확인합니다.

```bash
# embedding: 응답의 status가 ok인지 확인
curl -fsS http://127.0.0.1:8002/health

# LLM: 키를 화면이나 shell history에 남기지 않고 현재 shell에 잠시 로드
read -rsp "LLM API key: " LLM_API_KEY
printf '\n'
export LLM_API_KEY

# 인증된 응답에 "data" 목록이 있는지 확인
curl -fsS \
  -H "Authorization: Bearer ${LLM_API_KEY}" \
  http://127.0.0.1:8003/v1/models
unset LLM_API_KEY

# backend: 아래 JSON과 정확히 일치하는지 확인
curl -fsS http://127.0.0.1:8004/health
# {"status":"ok","service":"compasslm-backend"}
```

자동 포트 모드에서는 위 포트를 고정해서 사용하지 말고 `[READY]`/state URL을 probe
대상으로 사용하세요.

## 테스트와 공개 경계 검사

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/check_public_release.py --manifest PUBLIC_RELEASE_MANIFEST.txt
```

`PUBLIC_RELEASE_MANIFEST.txt`는 “저장소에서 발견한 모든 파일” 목록이 아니라 공개를
허용한 파일의 정확한 manifest입니다. 릴리스 검사기는 이 목록의 정렬, 누락, 파일
형식, 크기, 비밀 값과 로컬 경로를 검사합니다.

## 보안과 공개 범위

- 기본 호스트는 외부에 노출되지 않는 `127.0.0.1`입니다. 원격 공개는 방화벽,
  TLS, 접근 제어를 별도로 설계한 뒤에만 설정하세요.
- `replace-with-strong-secret`은 실행용 키가 아닙니다. 임베딩 서버/백엔드의
  `EMBEDDING_API_KEY`, LLM 서버/백엔드의 `LLM_API_KEY`를 각각 서로 맞는 강한
  값으로 교체하세요.
- `.env`, `runtime.env`, 비밀 키를 커밋하지 마세요.

다음 항목은 의도적으로 Git에 포함하지 않습니다.

- LLM 모델 가중치
- 임베딩/OCR 가중치
- 빌드된 LLM 런타임
- 사용자 데이터/업로드
- 로그
- SQLite 등 데이터베이스
- 오프라인 번들/패키지와 wheel
- 캐시, 가상환경, 생성 결과

라이선스는 [MIT](LICENSE)입니다. 모델, 데이터, 폰트, 외부 런타임에는 각 공급자의
별도 라이선스가 적용됩니다.
