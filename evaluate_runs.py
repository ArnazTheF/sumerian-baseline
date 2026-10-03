import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import sacrebleu
from rapidfuzz.distance import Levenshtein

TABLET_RE = re.compile(r"P\d{6}")
TAG_RE = re.compile(r"<[A-Z_]+>")
LINEBREAK = "\\n"
DEFAULT_PLACEHOLDERS = {"", "xxx"}


def read_tsv(path):
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def detect_task(name):
    parts = re.split(r"[_\-\s]+", name.lower())
    if any(p.startswith("ocr") for p in parts):
        return "ocr"
    if "translit" in parts or "transliteration" in parts:
        return "translit"
    if "mt" in parts or "translation" in parts:
        return "mt"
    return None


def squeeze(text):
    return re.sub(r"\s+", " ", text).strip()


def ocr_signs(text):
    text = TAG_RE.sub("", text).replace(LINEBREAK, "").replace("...", "")
    return re.sub(r"\s+", "", text)


def mt_normalize(text, lowercase):
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ord(ch) < 128)
    text = re.sub(r"[^\w\s]", "", text)
    text = squeeze(text)
    return text.lower() if lowercase else text


def has_loop(text, n=3, times=3):
    words = text.split()
    for i in range(len(words) - n * times + 1):
        chunk = words[i:i + n]
        if all(words[i + k * n:i + (k + 1) * n] == chunk for k in range(1, times)):
            return True
    return False


def is_degenerate(ref, hyp):
    if has_loop(hyp) and not has_loop(ref):
        return True
    return len(ref) >= 20 and len(hyp) > 2 * len(ref)


def cer_values(refs, hyps):
    dists = [Levenshtein.distance(r, h) for r, h in zip(refs, hyps)]
    per = [100 * d / max(len(r), 1) for d, r in zip(dists, refs)]
    total = sum(len(r) for r in refs)
    return dists, per, 100 * sum(dists) / max(total, 1)


def score(refs, hyps, use_cer, use_bleu=True):
    n = len(refs)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score if use_bleu else float("nan"),
        "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
        "exact": 100 * sum(r == h for r, h in zip(refs, hyps)) / n,
        "len_ratio": sum(len(h) for h in hyps) / max(sum(len(r) for r in refs), 1),
        "empty_pred": 100 * sum(not h.strip() for h in hyps) / n,
        "degenerate": 100 * sum(is_degenerate(r, h) for r, h in zip(refs, hyps)) / n,
    }
    if use_cer:
        _, per, corpus = cer_values(refs, hyps)
        per_series = pd.Series(per)
        out["cer_corpus"] = corpus
        out["cer_mean"] = per_series.mean()
        out["cer_median"] = per_series.median()
    return out


def build_variants(df, task, placeholders, lowercase):
    ref = df["reference"].map(squeeze)
    hyp = df["prediction"].map(squeeze)
    if task == "translit":
        return [("raw", ref, hyp, True, True, ref.str.len())], "raw"
    if task == "ocr":
        signs_ref = df["reference"].map(ocr_signs)
        signs_hyp = df["prediction"].map(ocr_signs)
        return [
            ("raw", ref, hyp, True, True, ref.str.len()),
            ("signs", signs_ref, signs_hyp, True, False, signs_ref.str.len()),
        ], "signs"
    keep = ~ref.str.lower().isin(placeholders)
    norm_ref = ref[keep].map(lambda s: mt_normalize(s, lowercase))
    norm_hyp = hyp[keep].map(lambda s: mt_normalize(s, lowercase))
    return [
        ("raw", ref, hyp, False, True, ref.str.split().str.len()),
        ("no_placeholder", ref[keep], hyp[keep], False, True, ref[keep].str.split().str.len()),
        ("normalized", norm_ref, norm_hyp, False, True, norm_ref.str.split().str.len()),
    ], "normalized"


def by_length(ref, hyp, units, use_cer, use_bleu, bins):
    frame = pd.DataFrame({"ref": ref, "hyp": hyp, "units": units})
    try:
        frame["bin"] = pd.qcut(frame["units"], bins, duplicates="drop")
    except ValueError:
        frame["bin"] = "all"
    rows = []
    for label, part in frame.groupby("bin", observed=True):
        row = {"length_bin": str(label)}
        row.update(score(list(part["ref"]), list(part["hyp"]), use_cer, use_bleu))
        rows.append(row)
    return rows


def worst_examples(df, ref, hyp, use_cer, k):
    if use_cer:
        _, per, _ = cer_values(list(ref), list(hyp))
        errors = pd.Series(per, index=ref.index)
        metric = "cer"
    else:
        errors = pd.Series(
            [100 - sacrebleu.sentence_chrf(h, [r]).score for r, h in zip(ref, hyp)],
            index=ref.index,
        )
        metric = "100-chrf"
    top = errors.sort_values(ascending=False).head(k).index
    out = df.loc[top, [c for c in ["id", "source"] if c in df.columns]].copy()
    out["reference"] = ref.loc[top]
    out["prediction"] = hyp.loc[top]
    out["error_metric"] = metric
    out["error"] = errors.loc[top].round(2)
    return out


def load_tablet_ids(args):
    ids = set()
    if args.train_ids:
        ids |= set(TABLET_RE.findall(Path(args.train_ids).read_text(encoding="utf-8")))
    if args.sumtablets is not None:
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("Для --sumtablets нужен пакет datasets: pip install datasets")
        for split in args.sumtablets or ["train"]:
            data = load_dataset("colesimmons/SumTablets", split=split)
            for value in data["id"]:
                ids |= set(TABLET_RE.findall(str(value)))
    return ids


def load_train_sources(path):
    if not path:
        return set()
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".csv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                            quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
        if "source" not in frame.columns:
            sys.exit(f"В {path} нет колонки source")
        values = frame["source"]
    else:
        values = path.read_text(encoding="utf-8").splitlines()
    return {squeeze(v) for v in values if v.strip()}


def seen_mask(df, tablet_ids, train_sources):
    if not tablet_ids and not train_sources:
        return None
    mask = pd.Series(False, index=df.index)
    if tablet_ids:
        found = df["id"].str.extract(r"(P\d{6})", expand=False)
        mask |= found.isin(tablet_ids)
    if train_sources and "source" in df.columns:
        mask |= df["source"].map(squeeze).isin(train_sources)
    return mask


def find_runs(root):
    for path in sorted(root.rglob("predictions_*.tsv")):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        task = detect_task(path.parent.parent.name)
        if task is None:
            print(f"Пропускаю {path}: не удалось определить задачу по имени папки", file=sys.stderr)
            continue
        yield path, task


def markdown_table(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if math.isnan(v) else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rounded(record):
    return {k: (None if math.isnan(v) else round(v, 2)) if isinstance(v, float) else v
            for k, v in record.items()}


def main():
    parser = argparse.ArgumentParser(description="Пересчёт метрик прогонов с нормализацией и разбивкой по длине")
    parser.add_argument("runs", nargs="?", default="runs", help="папка с прогонами")
    parser.add_argument("--out", help="куда писать отчёты (по умолчанию <runs>/eval)")
    parser.add_argument("--lowercase", action="store_true", help="для перевода сравнивать без учёта регистра")
    parser.add_argument("--placeholder", action="append", default=[], help="доп. заглушки в эталонах перевода")
    parser.add_argument("--train-ids", help="файл с P-номерами обучающих табличек для проверки утечки")
    parser.add_argument("--train-sources", help="txt или tsv с колонкой source из обучающей выборки")
    parser.add_argument("--sumtablets", nargs="*", metavar="SPLIT",
                        help="взять P-номера из сплитов SumTablets на Hugging Face (по умолчанию train)")
    parser.add_argument("--bins", type=int, default=4, help="число квантильных корзин по длине")
    parser.add_argument("--worst", type=int, default=20, help="сколько худших примеров сохранять")
    args = parser.parse_args()

    root = Path(args.runs)
    out_dir = Path(args.out) if args.out else root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    placeholders = DEFAULT_PLACEHOLDERS | {p.lower() for p in args.placeholder}
    tablet_ids = load_tablet_ids(args)
    train_sources = load_train_sources(args.train_sources)

    summary, length_rows, worst_frames = [], [], []
    for path, task in find_runs(root):
        df = read_tsv(path)
        missing = {"id", "reference", "prediction"} - set(df.columns)
        if missing:
            print(f"Пропускаю {path}: нет колонок {sorted(missing)}", file=sys.stderr)
            continue
        baseline = path.parent.name
        task_dir = path.parent.parent.name
        meta_path = path.with_name(path.name.replace("predictions_", "metrics_").replace(".tsv", ".json"))
        model = ""
        if meta_path.exists():
            model = json.loads(meta_path.read_text(encoding="utf-8")).get("baseline", "")
        base = {"task": task_dir, "baseline": baseline, "model": model}

        variants, main_variant = build_variants(df, task, placeholders, args.lowercase)
        seen = seen_mask(df, tablet_ids, train_sources)
        report = {**base, "variants": {}}

        for name, ref, hyp, use_cer, use_bleu, units in variants:
            record = {**base, "variant": name, **score(list(ref), list(hyp), use_cer, use_bleu)}
            if task == "ocr" and name == "raw":
                ref_lines = df["reference"].str.count(re.escape(LINEBREAK))
                hyp_lines = df["prediction"].str.count(re.escape(LINEBREAK))
                record["line_count_match"] = 100 * (ref_lines == hyp_lines).mean()
            summary.append(rounded(record))
            report["variants"][name] = rounded(record)

            if name != main_variant:
                continue
            for row in by_length(ref, hyp, units, use_cer, use_bleu, args.bins):
                length_rows.append(rounded({**base, "variant": name, **row}))
            worst = worst_examples(df, ref, hyp, use_cer, args.worst)
            worst.insert(0, "baseline", baseline)
            worst.insert(0, "task", task_dir)
            worst_frames.append(worst)
            if seen is not None:
                sub = seen.loc[ref.index]
                report["seen_in_train_pct"] = round(100 * sub.mean(), 2)
                for flag, label in [(True, "seen"), (False, "unseen")]:
                    part = sub[sub == flag].index
                    rec = {**base, "variant": f"{name}|{label}",
                           **score(list(ref.loc[part]), list(hyp.loc[part]), use_cer, use_bleu)}
                    summary.append(rounded(rec))
                    report["variants"][f"{name}|{label}"] = rounded(rec)

        (out_dir / f"{task_dir}__{baseline}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not summary:
        sys.exit(f"В {root} не найдено ни одного predictions_*.tsv")

    summary_df = pd.DataFrame(summary)
    lead = ["task", "baseline", "variant", "n"]
    summary_df = summary_df[lead + [c for c in summary_df.columns if c not in lead + ["model"]] + ["model"]]
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    length_df = pd.DataFrame(length_rows)
    length_df.to_csv(out_dir / "by_length.csv", index=False)
    pd.concat(worst_frames).to_csv(out_dir / "worst_examples.tsv", sep="\t", index=False,
                                   quoting=csv.QUOTE_NONE, escapechar="\\")

    text = ["Сводка", "======", "", markdown_table(summary_df.drop(columns=["model"])), "",
            "По длине эталона (основной вариант)", "===================================", "",
            markdown_table(length_df.drop(columns=["model"]))]
    (out_dir / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))
    print(f"\nОтчёты сохранены в {out_dir}")


if __name__ == "__main__":
    main()