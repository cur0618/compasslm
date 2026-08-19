# PaddleOCR-VL HPS 운영 가이드

PaddleOCR HPS는 대규모 GPU 서버에서 선택적으로 사용하는 배포 경로입니다. 공식
PaddleOCR HPS 배포 묶음과 `docker-compose.yml`을 이 디렉터리의 `runtime/`에
배치하거나, 묶음의 실제 위치를 `PADDLEOCR_HPS_DEPLOY_DIR`로 지정합니다. 공개
저장소에는 이미지, 모델 볼륨과 공식 배포 묶음이 포함되지 않습니다.

## 시작과 상태 확인

```bash
cp hps.env.example hps.env
./manage_hps.sh start
./manage_hps.sh ready
./manage_hps.sh logs -f
./manage_hps.sh stop
```

`start`는 Docker Compose 서비스를 시작하고 준비 상태가 될 때까지 기다립니다.
별도로 기다리려면 `./manage_hps.sh wait-ready`를 사용합니다. 기본 준비 상태 주소는
`http://127.0.0.1:8080/health/ready`이며, 관련 값은 `hps.env`와
`project-gpu/runtime.env`에서 운영 환경에 맞춥니다.

## 폐쇄망 반입

폐쇄망 서버에서는 준비된 이미지를 `docker save`로 내보내고, 공식 배포 묶음과 모델
볼륨을 SHA-256 목록과 함께 반입합니다. `manage_hps.sh start` 전에 해시를 검증하고
이미지를 불러옵니다.

```bash
./prepare_offline_bundle.sh /path/to/output-bundle
sha256sum -c /path/to/output-bundle/SHA256SUMS
docker load -i /path/to/output-bundle/paddleocr-hps-images.tar
```

## 백엔드 전환 기준

기본 공개 설정은 `PDF_OCR_BACKEND=ppocr_fast_v1`입니다. HPS 준비만으로 기본값을
바꾸지 말고, 고정 PDF 벤치마크와 품질 검증을 통과한 경우에만 시험 또는 운영
설정에서 `PDF_OCR_BACKEND=hps`를 사용합니다. HPS 장애 시 로컬 VL 경로로 전환하려면
`PDF_OCR_HPS_FALLBACK_TO_LOCAL=1`을 유지합니다.

고정 PDF 조정·검증은 다음 순서로 실행합니다.

```bash
../benchmark_pdf_ocr_hps_matrix.sh /path/to/reference.pdf
```
