#!/usr/bin/env python3
"""Проверка предсказаний и пересчёт метрик."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import statistics

from evaluate_collection import plans, read, sign_tokens, summarize


EDITORIAL_REFERENCE = re.compile(
    r"too fragmentary|no translation possible|not translated|untranslatable", re.I
)


def saved_prediction_path(value):
    path = Path(value)
    relocation = Path("legacy/relocation.json")
    if not path.exists() and relocation.exists():
        for record in json.loads(relocation.read_text())["files"]:
            if record["old"] == str(path):
                return Path(record["new"])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("legacy/technical_runs/new_data"))
    args = parser.parse_args()
    output = {"operation": "analysis_of_saved_predictions_only", "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "transformers", "tokenizers", "sacrebleu", "rapidfuzz")
    }, "evaluation_script_sha256": hashlib.sha256(
        Path("scripts/evaluate_collection.py").read_bytes()).hexdigest(), "tasks": {}}
    for task in ("mt", "translit", "ocr"):
        folder = args.root / task
        summary = json.loads((folder / "summary_full.json").read_text())
        records = read(folder / "predictions.jsonl")
        manifest=json.loads((folder/'manifest.json').read_text())
        expected, _ = plans(task,translit_fraction=manifest.get('sampling',{}).get('requested_fraction'),
                            control_limit=None if manifest.get('control_selection')=='all' else 100)
        actual_ids = [r["eval_id"] for r in records]
        assert len(actual_ids) == len(set(actual_ids)), f"Duplicate predictions: {task}"
        assert set(actual_ids) == {r["eval_id"] for r in expected}, f"Incomplete prediction set: {task}"
        assert all(r["status"] in ("ok", "input_exceeds_512_tokens") for r in records)
        for cohort in ("external", "legacy_control"):
            actual = summarize([r for r in records if r["cohort"] == cohort], task)
            saved = summary["cohorts"][cohort].copy()
            # Порядок суммирования влияет на погрешность времени.
            actual.pop("seconds", None)
            saved.pop("seconds", None)
            assert actual == saved, f"Metric mismatch: {task}/{cohort}"
        new = [r for r in records if r["cohort"] == "external"]
        complete = [r for r in new if r["status"] == "ok"]
        diagnostics = {"verified_checkpoint_records": len(records)}
        provenance_path=folder/'reused_predictions_provenance.json'
        if provenance_path.exists():
            provenance=json.loads(provenance_path.read_text())
            prior_path=saved_prediction_path(provenance.get('source',provenance.get('reused_from')))
            assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==provenance['sha256']
            previous=read(prior_path)
            current={r['eval_id']:r for r in records}
            for row in previous:
                saved=current[row['eval_id']]
                for field in ['src','tgt','prediction','status']:
                    assert row.get(field)==saved.get(field),f'Reused prediction changed: {task}/{row["eval_id"]}/{field}'
            diagnostics['reused_predictions']={
                'verified_unchanged':len(previous),'additional_predictions':len(records)-len(previous),
                'source':str(prior_path),
                'external':summarize([r for r in previous if r['cohort']=='external'],task),
                'legacy_control':summarize([r for r in previous if r['cohort']=='legacy_control'],task)}
        if task != "ocr":
            diagnostics["input_lengths"] = {}
            for cohort in ("external", "legacy_control"):
                members=[r for r in records if r['cohort']==cohort]
                lengths=sorted(len(r['src']) for r in members)
                info={"source_chars_median":statistics.median(lengths),
                      "source_chars_p90":lengths[int(.9*(len(lengths)-1))],
                      "source_chars_max":max(lengths)}
                if task=='translit':
                    info['cuneiform_signs_median']=statistics.median(r['input_sign_count'] for r in members)
                diagnostics['input_lengths'][cohort]=info
        if task == "mt":
            flagged = [r for r in new if EDITORIAL_REFERENCE.search(r["tgt"])]
            control = [r for r in records if r["cohort"] == "legacy_control"]
            with (folder / "editorial_reference_flags.jsonl").open("w", encoding="utf-8") as f:
                for row in flagged:
                    f.write(json.dumps({"id": row["id"], "source": row["source"],
                                        "reference": row["tgt"], "src_original": row.get("src_original", row["src"]),
                                        "reason": "editorial_comment_instead_of_translation",
                                        "inference_status": row["status"]}, ensure_ascii=False) + "\n")
            diagnostics.update(
                editorial_reference_ids=[r["id"] for r in flagged],
                legacy_references_with_xxx=sum(bool(re.search(r"\bxxx\b", r["tgt"], re.I)) for r in control),
                legacy_without_xxx=summarize([
                    r for r in control if not re.search(r"\bxxx\b", r["tgt"], re.I)], task),
                excluding_editorial_references=summarize(
                    [r for r in new if not EDITORIAL_REFERENCE.search(r["tgt"])], task),
                without_input_unk=summarize([r for r in new if not r.get("input_unk_count")], task),
                with_input_unk=summarize([r for r in new if r.get("input_unk_count")], task),
            )
        if task == "translit":
            diagnostics.update(
                input_x_damage_examples=sum(bool(re.search("[xX]", r["src"])) for r in complete),
                all_input_signs_dropped=sum(
                    r.get("input_sign_count", 0) > 0 and r.get("input_sign_count") == r.get("input_dropped_signs")
                    for r in complete),
                partly_or_fully_dropped=summarize([r for r in new if r.get("input_dropped_signs")], task),
                fully_representable=summarize([r for r in new if not r.get("input_dropped_signs")], task),
                no_dropped_glyphs_or_x_damage=summarize([
                    r for r in new if not r.get("input_dropped_signs") and not re.search("[xX]", r["src"])], task),
                short_inputs_1_10_signs=summarize([r for r in new if 1 <= r.get("input_sign_count", 0) <= 10], task),
                longer_inputs_11_plus_signs=summarize([r for r in new if r.get("input_sign_count", 0) > 10], task),
            )
        if task == "ocr":
            lengths = collections.Counter(len(sign_tokens(r["prediction"])) for r in complete)
            first = collections.Counter(
                next(iter(sign_tokens(r["prediction"])), "<EMPTY>") for r in complete)
            diagnostics.update(
                output_sign_length_counts=dict(sorted(lengths.items())),
                most_common_first_sign=first.most_common(10),
                reference_in_vocabulary=summarize([r for r in new if not r.get("reference_oov_signs")], task),
                first_sign_match_pct=round(100 * sum(
                    sign_tokens(r["prediction"])[:1] == sign_tokens(r["tgt"])
                    for r in complete) / max(1, len(complete)), 2),
            )
        output["tasks"][task] = {"summary": summary, "diagnostics": diagnostics}
        with (folder / "skipped_inputs.jsonl").open("w", encoding="utf-8") as f:
            for row in records:
                if row['status']!='ok':
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
        with (folder / "predictions_review.tsv").open("w", encoding="utf-8", newline="") as f:
            columns = ["eval_id", "cohort", "source", "status", "input", "reference", "prediction", "hit_generation_limit"]
            writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            for r in sorted(records, key=lambda r: r["eval_id"]):
                writer.writerow(dict(
                    eval_id=r["eval_id"], cohort=r["cohort"], source=r["source"], status=r["status"],
                    input=r.get("src_original", r.get("src", r.get("image_ref", r.get("image")))),
                    reference=r["tgt"], prediction=r.get("prediction", ""),
                    hit_generation_limit=r.get("hit_generation_limit", False)))
    (args.root / "analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for task, results in output["tasks"].items():
        print(task, json.dumps(results["summary"]["cohorts"], ensure_ascii=False))
    print("Verified all planned predictions and recomputed cohort metrics.")


if __name__ == "__main__":
    main()
