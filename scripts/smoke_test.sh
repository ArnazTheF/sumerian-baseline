set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
PY="python -m sumbase"

python tests/make_fixtures.py
$PY info
for task in mt translit ocr; do
  echo "=== $task: prepare + train + evaluate ==="
  $PY prepare  --config configs/smoke/$task.yaml
  $PY train    --config configs/smoke/$task.yaml
  $PY evaluate --config configs/smoke/$task.yaml --split valid
done
echo "=== memcheck (оценка видеопамяти) ==="
$PY memcheck --config configs/smoke/mt.yaml --budget-gb 12
echo "=== translation memory ==="
$PY baseline --config configs/smoke/mt.yaml --kind tm
echo "=== predict: текст и сквозной пайплайн ==="
$PY predict --run runs/smoke_mt --text "1(disz) udu niga"
$PY predict --pipeline --image tests/fixtures/ocr/images/T000.jpg \
  --ocr-run runs/smoke_ocr --translit-run runs/smoke_translit --mt-run runs/smoke_mt
echo "SMOKE TEST OK"
