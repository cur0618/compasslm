# 백엔드 소스 경계와 실행

백엔드 실행 코드의 정본은 저장소 루트의 `src/`입니다. 이 디렉터리는 별도의 소스
사본을 두는 배포 폴더가 아니며, 루트 `src/`를 이곳으로 복사하거나 동기화하지
않습니다.

일반 실행은 저장소 루트에서 통합 실행기를 사용합니다.

```bash
cd ~/compasslm
project-gpu/compass_up.sh
```

백엔드만 따로 진단할 때에는 선행 임베딩·LLM 서버가 준비된 상태에서 다음 런처를
사용합니다.

```bash
cd ~/compasslm
project-gpu/run_backend_api.sh
```

런처는 `project-gpu/main-backend/compassvenv`의 Python으로
`python -m uvicorn src.main:app`을 실행합니다. 설정은 서비스별 `.env`, `.env.auto`,
`project-gpu/runtime.env`, 실행 셸 환경변수 순서로 적용하고, 자동 포트 모드에서는
8004부터 사용 가능한 백엔드 포트를 선택합니다. 실제 주소는 `[READY] backend`
로그와 `COMPASS_PORT_STATE_FILE`이 가리키는 `ports.env`에서 확인합니다.

개발 중 자동 재적재가 필요할 때만
`API_RELOAD=1 project-gpu/run_backend_api.sh`를 사용하세요. API 키나 선택 포트를
이 문서의 명령에 직접 넣지 말고 `project-gpu/runtime.env`에서 관리합니다.
