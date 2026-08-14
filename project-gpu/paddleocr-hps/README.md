# PaddleOCR-VL HPS runtime

Place the official PaddleOCR HPS deployment bundle at `runtime/`, including
`docker-compose.yml`, or set `PADDLEOCR_HPS_DEPLOY_DIR` to that directory.

```bash
cp hps.env.example hps.env
./manage_hps.sh start
./manage_hps.sh ready
./manage_hps.sh stop
```

For an offline server, export the prepared images with `docker save`, copy the
official deployment bundle and model volumes with their SHA-256 manifest, then
run `docker load` before `manage_hps.sh start`.

```bash
./prepare_offline_bundle.sh /path/to/output-bundle
sha256sum -c /path/to/output-bundle/SHA256SUMS
docker load -i /path/to/output-bundle/paddleocr-hps-images.tar
```

The application remains on `PDF_OCR_BACKEND=local` until the H100 acceptance
benchmark passes. Set `PDF_OCR_BACKEND=hps` only for the HPS benchmark or after
promotion.

Run the fixed-PDF tuning sequence with:

```bash
../benchmark_pdf_ocr_hps_matrix.sh /path/to/reference.pdf
```
