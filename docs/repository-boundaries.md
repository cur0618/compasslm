# 공개 저장소 경계

CompassLM 공개 저장소의 경계는 `PUBLIC_RELEASE_MANIFEST.txt`에 적힌 정확한
파일 목록이다. 디렉터리 전체를 공개 대상으로 간주하지 않으며, manifest에 없는
로컬 파일은 Git에 올리지 않는다.

## 검증 기준

- manifest는 자기 자신을 포함하고 정렬되어 있으며 중복이 없어야 한다.
- 공개 파일은 소스, 실행 스크립트, 안전한 예시 설정, 문서와 공개 테스트로
  제한한다.
- 모델·tokenizer 가중치, 네이티브 런타임, 실제 환경 설정, 사용자 데이터,
  인덱스·DB·로그·캐시와 오프라인 패키지는 공개하지 않는다.
- 모든 manifest 파일의 경로, 형식, 크기와 텍스트 내용은 다음 명령으로 검사한다.

```bash
python3 scripts/check_public_release.py --manifest PUBLIC_RELEASE_MANIFEST.txt
```

## 변경 절차

공개 파일을 추가하거나 제거할 때는 manifest를 먼저 정렬·중복 없는 exact
allowlist로 갱신한다. 그 다음 공개 guard와 테스트를 실행하고,
`git ls-files` 결과가 manifest와 정확히 같은지 확인한다. manifest 밖의 로컬
운영 자료나 도구는 공개 저장소에서 사용할 수 있는 구성으로 문서화하지 않는다.
