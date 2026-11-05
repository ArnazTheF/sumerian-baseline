#!/usr/bin/env python3
"""Проверка покрытия токенизатора и уникальности ID."""
from __future__ import annotations

import collections
import json
from pathlib import Path
import re

from evaluate_collection import cuneiform_chars, load_tok, read
from sumbase.corpus.build import digest


def audit(rows, field):
    tokenizer = load_tok("colesimmons/SumerianGlyphTokenizer_Roberta")
    by_source = collections.defaultdict(collections.Counter)
    missing = collections.Counter()
    for row in rows:
        text = row[field]
        original = collections.Counter(cuneiform_chars(text))
        ids = tokenizer(text, truncation=False)["input_ids"]
        encoded = collections.Counter(cuneiform_chars("".join(tokenizer.convert_ids_to_tokens(ids))))
        lost = original - encoded
        missing.update(lost)
        counts = by_source[row.get("source", "unknown")]
        counts.update(examples=1, signs=sum(original.values()), dropped_signs=sum(lost.values()),
                      examples_with_dropped_signs=bool(lost),
                      all_signs_dropped=bool(original) and original == lost,
                      examples_with_x_damage=bool(re.search("[xX]", text)),
                      no_dropped_signs_or_x_damage=not lost and not re.search("[xX]", text))
    total = sum(by_source.values(), collections.Counter())
    return {"total": dict(total), "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
            "unsupported_sign_counts": dict(missing.most_common())}


def main():
    translit_rows=read("data/collection/ready/translit/test.jsonl")
    identities=collections.Counter(r['id'] for r in translit_rows)
    revised_ids={digest(r['document']+r['line_id'] if r['line_id'] else
                        '\0'.join([r['document'],r['src'],r['tgt']]))[:24] for r in translit_rows}
    assert len(revised_ids)==len(translit_rows), 'The revised exporter still produces ID collisions'
    result = {"operation": "tokenizer_only_no_model_inference_or_training",
              "revision": "75fadd99143f2d2fc690c285b7228d262c524be1",
              "translit_identity_audit": {"rows":len(translit_rows),"legacy_unique_ids":len(identities),
                  "colliding_legacy_ids":sum(n>1 for n in identities.values()),
                  "rows_with_colliding_legacy_ids":sum(n for n in identities.values() if n>1),
                  "revised_exporter_unique_ids":len(revised_ids),"frozen_data_changed":False},
              "translit_all_external_candidates": audit(translit_rows, "src"),
              "ocr_test_references": audit(read("data/collection/ready/ocr/test.jsonl"), "tgt")}
    path = Path("legacy/technical_runs/new_data/tokenization_audit.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for task in ("translit_all_external_candidates", "ocr_test_references"):
        print(task, result[task]["total"])


if __name__ == "__main__":
    main()
