#!/usr/bin/env python3
"""Экспорт сохранённых результатов для просмотра без запуска моделей."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TECHNICAL = REPO / "legacy/technical_runs"
RESULTS = REPO / "runs"
TASKS = {
    "mt": ("translation", "Перевод", "praeclarum/cuneiform"),
    "translit": ("transliteration", "Транслитерация", "colesimmons/sumerian-transliteration"),
    "ocr": ("image_recognition", "Распознавание", "colesimmons/lineart-ocr"),
}
METRIC_COLUMNS = [
    "task",
    "method",
    "text_variant",
    "evaluated",
    "skipped",
    "BLEU",
    "chrF",
    "CER_percent",
    "exact_match_percent",
    "note",
]


def write_metrics(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(task, method, variant, values, skipped=0, note=""):
    return dict(
        zip(
            METRIC_COLUMNS,
            [
                TASKS[task][1],
                method,
                variant,
                values.get("n", ""),
                skipped,
                values.get("bleu", ""),
                values.get("chrf", ""),
                values.get("cer", ""),
                values.get("exact_match", ""),
                note,
            ],
        )
    )


def export_original():
    output = RESULTS / "original_data"
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for task, (name, _, model) in TASKS.items():
        source_task = "ocr_lineart" if task == "ocr" else task
        for method in ("zeroshot", "tm") if task != "ocr" else ("zeroshot",):
            folder = TECHNICAL / "original_data" / f"nogpu_{source_task}" / f"baseline_{method}"
            suffix = "memory_predictions" if method == "tm" else "predictions"
            shutil.copyfile(folder / "predictions_test.tsv", output / f"{name}_{suffix}.tsv")
            values = json.loads((folder / "metrics_test.json").read_text())
            note = "Исходная метрика запуска; OCR-тесты несопоставимы" if task == "ocr" else ""
            metrics.append(
                metric_row(
                    task,
                    "translation memory" if method == "tm" else model,
                    "исходный",
                    values,
                    note=note,
                )
            )
    write_metrics(output / "metrics.csv", metrics)


def export_new():
    output = RESULTS / "new_data"
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for task, (name, _, model) in TASKS.items():
        folder = TECHNICAL / "new_data" / task
        summary = json.loads((folder / "summary_full.json").read_text())["cohorts"]["external"]
        columns = [
            "id",
            "source",
            "status",
            "input",
            "reference",
            "prediction",
            "hit_generation_limit",
        ]
        with (
            (folder / "predictions_review.tsv").open(encoding="utf-8", newline="") as source,
            (output / f"{name}_predictions.tsv").open("w", encoding="utf-8", newline="") as target,
        ):
            writer = csv.DictWriter(target, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            count = 0
            for row in csv.DictReader(source, delimiter="\t"):
                if row["cohort"] == "external":
                    writer.writerow(
                        {key: row["eval_id"] if key == "id" else row[key] for key in columns}
                    )
                    count += 1
            assert count == summary["planned"], (task, count, summary["planned"])
        skipped = sum(summary["skipped"].values())
        if task == "ocr":
            values = {
                "n": summary["evaluated"],
                "cer": summary["sign_cer"],
                "exact_match": summary["sign_exact"],
            }
            metrics.append(
                metric_row(
                    task,
                    model,
                    "по знакам",
                    values,
                    skipped,
                    "Одиночные знаки; несопоставимо с OCR целых прорисовок",
                )
            )
        else:
            for key, label in (("raw", "исходный"), ("normalized", "нормализованный")):
                metrics.append(metric_row(task, model, label, summary[key], skipped))
    write_metrics(output / "metrics.csv", metrics)
    shutil.copyfile(
        TECHNICAL / "new_data/reports/SUMERIAN_MODEL_EVALUATION_FULL_TRANSLIT_2026-11-05.docx",
        output / "evaluation_report.docx",
    )


def main():
    export_original()
    export_new()
    print("Результаты сохранены в runs/original_data и runs/new_data.")


if __name__ == "__main__":
    main()
