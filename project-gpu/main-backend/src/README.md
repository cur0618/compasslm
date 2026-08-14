현재 `main-backend`는 루트 `src/` 코드를 기준으로 실행한다.

권장 실행:
```bash
cd \compasslm
source project-gpu/main-backend/compassvenv/bin/activate
export EMBEDDING_PROVIDER=api
export EMBEDDING_API_URL=http://127.0.0.1:8001
export EMBEDDING_API_KEY=replace-with-strong-secret
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

향후 배포 단계에서 루트 `src/`를 이 디렉터리로 동기화해 독립 배포 단위로 분리한다.
