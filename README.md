# CompassLM

CompassLM은 로컬 LLM, 임베딩, OCR을 결합해 문서를 색인하고 검색 근거와 인용을
포함한 답변을 만드는 한국어 중심 문서 RAG 애플리케이션입니다. 계정별 지식공간,
비동기 문서 처리, 구조·표 검색, Wiki answer memory와 Ontology-RAG 보조 검색을
하나의 FastAPI 웹 UI에서 운영합니다.

이 공개 저장소는 다른 컴퓨터에서 구조와 설정을 재현할 수 있도록 소스, 테스트,
실행 스크립트와 안전한 예제 설정만 제공합니다. 모델 가중치와 운영 데이터가 포함된
완제품 배포본은 아닙니다.

## 주요 기능

- TXT, XLSX, HWPX, PDF 업로드와 형식별 구조 보존
- 로그인 사용자별 계정·세션·지식공간·채팅·업로드 격리
- OCR과 색인·임베딩을 나눈 비동기 2단계 업로드 및 세부 진행률 표시
- SQLite, FTS, 벡터 검색을 결합한 하이브리드 retrieval
- 표 행·정의 블록·문서 제목 구조를 활용한 숫자 및 표 질의 보강
- PydanticAI 기반 질문 분석, 검색 도구 호출, 답변 검증과 제한적 재시도
- PDF 페이지, XLSX 시트·행, TXT 줄 범위를 보여 주는 사용자용 인용
- 검증된 답변을 재사용하는 Wiki answer memory와 Ontology-RAG 보조 검색
- tmux 기반 통합 기동·상태·로그·종료와 운영 데이터 보존 정책

## 구성과 데이터 흐름

백엔드/UI가 요청을 받아 임베딩과 LLM을 각각 호출하는 orchestrator입니다.

```text
사용자/업로드 → backend/UI orchestrator
backend → embedding API → 색인·질의 벡터
local data/SQLite/index → retrieval → 검색 근거 → backend
backend → llama.cpp LLM → 답변 생성 → backend
backend → 인용(citations) 포함 응답 → 사용자/UI
```

문서와 질문은 다음 순서로 처리됩니다.

1. 사용자가 로그인하고 계정별 지식공간을 선택하거나 만듭니다.
2. 백엔드가 확장자와 파일 signature를 확인하고 upload job을 저장합니다.
3. TXT·XLSX·HWPX는 빠른 처리 lane으로, PDF는 OCR lane으로 분리됩니다.
4. PDF는 PaddleOCR 기반 처리를 우선하고 필요한 경우 PyMuPDF 텍스트를 보조로
   사용합니다. 완료된 OCR payload와 색인 commit 단계는 별도로 관리됩니다.
5. 문서를 청킹하고 임베딩 API로 벡터화한 뒤 SQLite/FTS/vector index와
   구조·표·Wiki·Ontology 파생 정보를 갱신합니다.
6. 질문 분석, query rewrite, retrieval과 조건부 rerank로 근거 후보를 모읍니다.
7. llama.cpp LLM이 근거 안에서 답변하고 validator가 내용·인용 형식을 확인합니다.
8. UI는 내부 문서 번호 대신 파일명과 페이지·시트·행·줄 위치를 표시합니다.

같은 지식공간에 같은 원본 파일명을 다시 올리면 하나의 논리 문서로 교체하며,
chunk·vector·FTS·Wiki·Ontology 파생 정보도 함께 정리합니다. 업로드와 실행 중
생성되는 SQLite/index는 로컬 `data/`에만 남고 Git에 포함되지 않습니다.

주요 소스 트리는 다음과 같습니다.

```text
src/
  main.py                         FastAPI, 인증, 지식공간, 업로드, 채팅 API
  rag.py                          문서 ingest, SQLite/FTS/vector 검색과 인덱스
  pdf_ocr.py                      PDF OCR, worker lifecycle, fallback과 계측
  compass_ai/                     질문 분석·도구 호출·답변 생성
  auth_store.py                   계정, 세션, 사용자별 지식공간 소유권
  upload_job_store.py             비동기 upload job 영속화
  upload_progress.py              OCR·색인 단계별 진행률과 heartbeat
  kb_engine_registry.py           사용 중인 KB engine lease와 안전한 eviction
  persistence_retention.py        DB row·로그·JSONL 보존 정책
  document_structure.py           HWPX/XLSX 구조 metadata와 임베딩 텍스트
  table_facts.py                  표 행·정의 블록 의미화
  wiki_* / ontology_*             답변 memory, Wiki, Ontology 보조 검색
  static/ / templates/            로그인·업로드·채팅·근거 UI
project-gpu/
  compass_up.sh                   embedding → LLM → backend 통합 기동
  compass_status.sh               프로세스·포트·health·proxy 상태 확인
  compass_logs.sh                 최근 runtime 로그 확인
  compass_down.sh                 backend → LLM → embedding 순서 종료
  run_*                           서비스별 개별 실행기
  setup_gpu_track.sh              온라인·오프라인 Python 환경 설치
  check_gpu_assets.sh             모델·runtime·패키지 자산 검사
  runtime.env.example             공개 공통 설정 기준
  embedding-gpu-server/           임베딩 FastAPI 서비스
  main-backend/                   백엔드 환경과 LLM/OCR 자산 기준 경로
scripts/
  create_admin_user.py            최초 관리자 생성과 legacy KB 등록
  prune_runtime_logs.py           runtime 로그 보존 정책 적용
  check_public_release.py         공개 manifest와 민감정보 경계 검사
tests/                             기능·운영·공개 릴리스 계약 테스트
docs/MODELS_AND_ASSETS.md          모델 형식, 배치 경로와 설정
docs/repository-boundaries.md      공개·비공개 저장소 경계
PUBLIC_RELEASE_MANIFEST.txt        공개 파일의 정확한 허용 목록
```

## 준비 사항과 설치

- Python 3.11+
- Bash와 tmux를 사용할 수 있는 Linux
- GPU 경로는 NVIDIA GPU, 호환 드라이버와 CUDA 필요
- 저장 공간은 모델과 로컬 인덱스 크기를 별도로 고려

저장소를 받은 뒤 공통 설정을 먼저 복사합니다.

```bash
git clone https://github.com/cur0618/compasslm.git
cd compasslm
cp project-gpu/runtime.env.example project-gpu/runtime.env
```

PaddlePaddle runtime 종류와 CUDA track을 선택한 뒤 서비스 실행에 사용하는 두
가상환경을 설치 스크립트로 준비합니다.

```bash
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

서비스 실행기는 낮은 우선순위부터 다음 순서로 설정을 읽습니다.

```text
서비스별 `.env` → 서비스별 `.env.auto` → project-gpu/runtime.env → 실행 셸 환경변수
```

뒤에 읽은 값이 앞선 값을 덮어씁니다. `runtime.env.example`을 복사한
`project-gpu/runtime.env`가 공통 운영값의 기준입니다. 설치 스크립트가 만드는
`.env.auto`는 직접 수정하지 말고, 일시적 override는 실행 셸 환경변수에
지정하세요. 서비스별 `.env`는 낮은 우선순위의 호환 기본값으로만 사용됩니다.

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

### 최초 관리자 생성

백엔드 설치 후 대화형으로 관리자 비밀번호를 입력합니다.

```bash
project-gpu/main-backend/compassvenv/bin/python \
  scripts/create_admin_user.py --login admin
```

이 명령은 `data/app.sqlite`에 관리자를 만들고 기존 `data/kb/`가 있으면 해당 legacy
지식공간을 관리자 소유로 등록합니다. 변경 대상을 먼저 확인하려면 `--dry-run`을
추가하세요. 일반 사용자는 브라우저의 계정 만들기 화면에서 등록할 수 있습니다.

### 통합 실행 권장 방식

권장 방식은 tmux 통합 실행기입니다.

```bash
chmod +x project-gpu/*.sh project-gpu/embedding-gpu-server/*.sh
project-gpu/compass_up.sh
```

통합 실행기는 자산 검사 후 `embedding → LLM → backend` 순서로 시작하며, 앞선
서비스의 health probe가 준비된 뒤에만 다음 서비스를 실행합니다. 자동 포트가
선택되면 runtime state와 `logs/runtime/.../startup_summary.json`에 실제 URL과 PID가
기록됩니다.

운영 명령은 다음과 같습니다.

```bash
project-gpu/compass_status.sh
project-gpu/compass_status.sh --json
project-gpu/compass_logs.sh backend -f
project-gpu/compass_down.sh
tmux attach -t compasslm
```

기존 세션을 정상 종료하고 다시 시작하려면
`project-gpu/compass_up.sh --restart`를 사용합니다.

### 개별 실행 대안

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

자동 포트 모드에서는 위 포트를 고정해서 사용하지 말고
`project-gpu/compass_status.sh`와 `[READY]` 로그의 URL을 probe 대상으로 사용하세요.

## 운영과 데이터 보존

- 완료 upload job, 만료 로그인 session, RAG 로그, runtime 로그와 운영 JSONL은
  기간·건수·크기 기준으로 제한적으로 정리됩니다.
- 처리 중인 upload job과 사용 중인 KB engine은 정리·eviction 대상에서
  보호됩니다.
- `project-gpu/collect_debug_bundle.sh`는 최근 시작 요약, 서비스 로그, answer·retrieval
  진단과 신고된 근거 문제를 하나의 진단 묶음으로 수집합니다.
- 기본 Wiki answer memory는 검증된 답변을 빠르게 재사용합니다. Wiki page workflow와
  일반 검색 boost는 설정으로 별도 활성화합니다.
- Ontology-RAG는 추출된 사실과 관계를 보조 검색에 사용하며 rebuild job, 검토,
  publish/archive API를 제공합니다.

## 테스트와 공개 경계 검사

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/check_public_release.py --manifest PUBLIC_RELEASE_MANIFEST.txt
```

`PUBLIC_RELEASE_MANIFEST.txt`는 “저장소에서 발견한 모든 파일” 목록이 아니라 공개를
허용한 파일의 정확한 manifest입니다. 릴리스 검사기는 이 목록의 정렬, 누락, 파일
형식, 크기, 비밀 값과 로컬 경로를 검사합니다.

## GitHub와 공공 GitLab 운영

GitHub를 기준 저장소로 사용합니다. `main`을 포함한 GitHub 브랜치와 태그 변경은
GitHub Actions의 `.github/workflows/mirror-to-aigov-gitlab.yml`을 통해 공공 GitLab에
자동 반영됩니다. 이 동기화는 기존 원격 브랜치나 태그를 삭제하지 않는 push
방식입니다.

공공 GitLab은 소스 열람용 mirror이므로 GitLab Auto DevOps는 동기화에 필요하지
않습니다. 별도의 GitLab build/test/deploy를 운영하려는 경우에만 활성 runner와
프로젝트 전용 CI 설정을 추가하세요.

## 보안과 공개 범위

- 기본 호스트는 외부에 노출되지 않는 `127.0.0.1`입니다. 원격 공개는 방화벽,
  TLS, 접근 제어를 별도로 설계한 뒤에만 설정하세요.
- `replace-with-strong-secret`은 실행용 키가 아닙니다. 임베딩 서버/백엔드의
  `EMBEDDING_API_KEY`, LLM 서버/백엔드의 `LLM_API_KEY`를 각각 서로 맞는 강한
  값으로 교체하세요.
- `.env`, `.env.auto`, `runtime.env`, 비밀 키를 커밋하지 마세요.

다음 항목은 의도적으로 Git에 포함하지 않습니다.

- LLM 모델 가중치
- 임베딩/OCR 가중치
- 빌드된 LLM 런타임
- 사용자 데이터/업로드
- 로그
- SQLite 등 데이터베이스
- 오프라인 번들/패키지와 wheel
- 캐시, 가상환경, 생성 결과

자세한 공개 경계는
[`docs/repository-boundaries.md`](docs/repository-boundaries.md)를 참고하세요.

## 라이선스

소스 라이선스는 [MIT](LICENSE)입니다. 모델, 데이터, 폰트와 외부 런타임에는 각
공급자의 별도 라이선스가 적용됩니다.
