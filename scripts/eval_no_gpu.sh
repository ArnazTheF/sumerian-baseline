set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
PY="python -m sumbase"
TASKS="${TASKS:-mt translit ocr_lineart}"

$PY info
status=0
run() { echo; echo "=== $* ==="; "$@" || { echo "!!! ШАГ НЕ УДАЛСЯ: $*"; status=1; }; }
for t in $TASKS; do
  cfg="configs/nogpu/$t.yaml"
  run $PY prepare  --config "$cfg"
  [ "$t" != "ocr_lineart" ] && run $PY baseline --config "$cfg" --kind tm
  run $PY baseline --config "$cfg" --kind zeroshot
done

echo; echo "=== Итог ==="
for f in runs/nogpu_*/baseline_*/metrics_test.json; do
  [ -f "$f" ] && python - "$f" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
keys = [k for k in ("bleu", "chrf", "cer", "n") if k in m]
print(f"{sys.argv[1]:<60}", "  ".join(f"{k}={m[k]}" for k in keys))
PY
done
exit $status
