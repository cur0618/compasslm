# CompassLM Local Fonts

이 폴더의 폰트는 폐쇄망 서버에서도 동일한 UI를 유지하기 위한 로컬 정적 자산입니다.

- `compass-display.woff2`
  브랜드명, KB 제목, 배지, 주요 버튼, 라벨에 쓰는 display font
- `compass-body.woff2`
  채팅 본문, 파일 목록, 설명문, 입력창에 쓰는 body font
- `compass-mono.woff2`
  citation 번호와 정밀 메타 표시에 쓰는 mono font

출처

- `compass-display.woff2`: Pretendard GOV Variable 배포본
- `compass-body.woff2`: Pretendard Variable 배포본
- `compass-mono.woff2`: JetBrains Mono Regular Webfont 배포본

운영 규칙

- 외부 CDN이나 원격 폰트 링크는 사용하지 않습니다.
- 파일 이름은 위의 canonical 이름을 유지합니다.
- 교체가 필요하면 동일 역할의 `woff2` 파일만 덮어쓰고, `style.css`의 경로와 토큰 이름은 유지합니다.
- 폐쇄망 반입 시 이 폴더 전체를 함께 배포해야 합니다.
