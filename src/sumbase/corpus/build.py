from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sqlite3
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from .acquire import dump
from .parsers import (atf_documents, etcsl_documents, ebl_documents, bdtns_documents, fingerprint, loose_fingerprint,
                      oracc_documents, sumtablets_documents, ws, canonical_work_id)


def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pack(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), 6)


def unpack(value):
    return json.loads(zlib.decompress(value))


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        root = key
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while key != root:
            key, self.parent[key] = self.parent[key], root
        self.parent.setdefault(root, root)
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            a, b = sorted((a, b))
            self.parent[b] = a


def connect(root):
    db = sqlite3.connect(root / "corpus.sqlite3")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def catalogue(path):
    result = {}
    fields = ("designation", "collection", "museum_no", "composite_id", "genre", "subgenre", "period",
              "provenience", "language", "photo_up", "lineart_up", "author", "atf_source", "translation_source", "date_updated")
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tid = row.get("id_text", "") or row.get("id", "")
            if tid.isdigit():
                tid = f"P{int(tid):06d}"
            result[tid] = {k: row[k] for k in fields if row.get(k)}
    return result


def build(root: Path):
    raw = root / "raw"
    out = root / "exports"
    out.mkdir(parents=True, exist_ok=True)
    temporary = root / "corpus.build.sqlite3"
    if temporary.exists():
        temporary.unlink()
    db = sqlite3.connect(temporary)
    db.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE editions(uid TEXT PRIMARY KEY, source TEXT, text_id TEXT, group_id TEXT, data BLOB);
        CREATE TABLE identities(uid TEXT, alias TEXT, PRIMARY KEY(uid,alias));
        CREATE TABLE segments(id TEXT PRIMARY KEY, doc_uid TEXT, source TEXT, src TEXT, tgt TEXT,
                              src_key TEXT, tgt_key TEXT, loose_key TEXT, data BLOB);
        CREATE TABLE exposure(key TEXT, kind TEXT, PRIMARY KEY(key,kind));
        CREATE TABLE images(id TEXT PRIMARY KEY, tablet_id TEXT, group_id TEXT, sha256 TEXT, data BLOB);
        CREATE TABLE excluded_documents(uid TEXT PRIMARY KEY, reason TEXT);
    """)
    uf = UnionFind()
    stats = Counter()
    errors = []
    museum_index = defaultdict(set)

    def museum_key(value):
        value = re.sub(r"\b0+(\d)", r"\1", value.upper())
        return re.sub(r"[^A-Z0-9]", "", value)

    def ingest(records):
        for r in records:
            uid = r["uid"]
            aliases = sorted(set(r.get("aliases", []) + r.get("related", []) + [uid]))
            aliases = [canonical_work_id(a) if a.startswith("etcsl:") and re.fullmatch(r"[\d.]+", a[6:]) else a for a in aliases]
            metadata = r.get("metadata", {})
            if metadata.get("museum_no"):
                museum_matches = museum_index.get(museum_key(metadata["museum_no"]), set())
                if len(museum_matches) == 1:
                    aliases.extend(museum_matches)
            if "fake" in str(metadata.get("genre", "")).lower():
                r.setdefault("quality_flags", []).append("catalogued_modern_fake")
            bid = str(metadata.get("bdtns_id", "")).strip()
            if bid.isdigit():
                aliases.append("bdtns:" + str(int(bid)))
            # Нормализуем ведущие нули в ссылках на разделы ETCSL.
            for field in ("subgenre", "designation", "primary_publication", "chap", "etcsl_id", "etcsl"):
                for work in re.findall(r"(?i)ETCSL\s*:?\s*(\d+(?:\.\d+)+)", str(metadata.get(field, ""))):
                    aliases.append(canonical_work_id(work))
            for reference in metadata.get("witness_references", []):
                for match in re.finditer(r"\b(?:CBS|AO|Ni|HS|BM|VAT|IM|YBC|NBC|N|UM|U)\s+\d+(?:[-,]\d+)*[a-z]?", reference):
                    ids = museum_index.get(museum_key(match.group()), set())
                    if len(ids) == 1:
                        aliases.extend(ids)
            # Композит и его рукописи объединяем до разбиения.
            composite = r.get("metadata", {}).get("composite_id", "")
            aliases += re.findall(r"Q\d{6}", composite)
            if len(fingerprint(r.get("transliteration", ""))) >= 80:
                aliases.append("text:" + digest(loose_fingerprint(r["transliteration"])))
            for a in aliases:
                uf.union(uid, a)
            segments = r.pop("segments", [])
            inserted = db.execute("INSERT OR IGNORE INTO editions VALUES(?,?,?,NULL,?)",
                                  (uid, r["source"], r["text_id"], pack(r))).rowcount
            if not inserted:
                continue
            if "fake" in str(metadata.get("genre", "")).lower():
                db.execute("INSERT OR IGNORE INTO excluded_documents VALUES(?,?)", (uid, "catalogued_modern_fake"))
            db.executemany("INSERT OR IGNORE INTO identities VALUES(?,?)", ((uid, a) for a in aliases))
            stats[r["source"]] += 1
            for n, s in enumerate(segments):
                src, tgt = s["src"].strip(), s["tgt"].strip()
                if not src or not tgt:
                    continue
                key = uid + ":" + s.get("id", str(n))
                db.execute("INSERT OR IGNORE INTO segments VALUES(?,?,?,?,?,?,?,?,?)",
                           (key, uid, r["source"], src, tgt, digest(fingerprint(src)),
                            digest(ws(tgt).casefold()), digest(loose_fingerprint(src)), pack(s)))
            if r["source"] == "sumtablets":
                # Модель могла видеть любой опубликованный split.
                for line in re.split(r"<SURFACE>|<COLUMN>|\n", r["transliteration"]):
                    if ws(line):
                        db.execute("INSERT OR IGNORE INTO exposure VALUES(?,?)",
                                   (digest(loose_fingerprint(line)), "sumtablets_published"))
            if sum(stats.values()) % 5000 == 0:
                db.commit()

    meta = catalogue(raw / "cdli/cdli_cat.csv")
    for tid, metadata in meta.items():
        if metadata.get("museum_no"):
            museum_index[museum_key(metadata["museum_no"])].add(tid)
    print(f"CDLI catalogue: {len(meta):,} objects", flush=True)
    ingest(atf_documents(raw / "cdli/cdliatf_unblocked.atf", "cdli", "CDLI-terms-see-source", meta))
    print(f"CDLI Sumerian: {stats['cdli']:,}", flush=True)
    mtaac_atf = raw / "mtaac/dataset/original/sumerian_translated.atf"
    if mtaac_atf.exists():
        ingest(atf_documents(mtaac_atf, "mtaac_atf", "MIT-repository; underlying-CDLI-terms", meta))
    seen = set()
    # При пересечениях предпочитаем исходные проекты агрегатам ePSD/CDLI.
    paths = sorted((raw / "oracc").glob("*.zip"), key=lambda p: (p.stem.startswith(("epsd2", "cdli")), p.name))
    for path in paths:
        try:
            before = sum(stats.values())
            ingest(oracc_documents(path, seen, raw / "translations"))
            print(f"parsed {path.name}: +{sum(stats.values())-before:,} editions", flush=True)
        except Exception as exc:
            errors.append({"source": str(path), "error": repr(exc)})
            print(f"PARSER ERROR {path}: {exc}", flush=True)
    if (raw / "etcsl/mirror.zip").exists():
        ingest(etcsl_documents(raw / "etcsl/mirror.zip"))
    if (raw / "ebl-fragments.json").exists():
        ingest(ebl_documents(raw / "ebl-fragments.json"))
    ingest(bdtns_documents(raw / "bdtns"))
    for path in sorted((raw / "sumtablets").glob("*.csv")):
        ingest(sumtablets_documents(path))
    # Пары без ID документов нельзя включать в тест по независимым объектам.
    ingest_mtaac_pairs(raw, ingest, db)
    db.commit()
    print("Grouping duplicate editions and manuscript/composite families", flush=True)
    db.executemany("UPDATE editions SET group_id=? WHERE uid=?",
                   ((digest(uf.find(uid))[:24], uid) for (uid,) in db.execute("SELECT uid FROM editions").fetchall()))
    db.executescript("""
        CREATE INDEX editions_group ON editions(group_id);
        CREATE INDEX editions_text ON editions(text_id);
        CREATE INDEX identities_alias ON identities(alias);
        CREATE INDEX segments_doc ON segments(doc_uid);
        CREATE INDEX segments_loose ON segments(loose_key);
    """)
    db.commit()
    dump(root / "parser-errors.json", errors)
    dump(root / "source-counts.json", dict(stats))
    index_photos(raw, db)
    index_maicubeda(raw, db, root)
    db.commit()
    db.close()
    temporary.replace(root / "corpus.sqlite3")
    export(root)
    audit(root)


def ingest_mtaac_pairs(raw, ingest, db):
    base = raw / "mtaac/dataset"
    # mono и обратные переводы не используем как эталоны.
    for path in sorted(base.rglob("*.sum")):
        target = path.with_suffix(".eng")
        if not target.exists():
            continue
        src = path.read_text(encoding="utf-8").splitlines()
        tgt = target.read_text(encoding="utf-8").splitlines()
        if len(src) != len(tgt):
            raise ValueError(f"MTAAC unaligned: {path}: {len(src)} != {len(tgt)}")
        relative = str(path.relative_to(base))
        # ID табличек отсутствуют; группируем по имени файла.
        record = {"uid": "mtaac_pairs:" + relative, "source": "mtaac_pairs", "text_id": relative,
                  "aliases": [], "title": relative, "metadata": {}, "language": "sux",
                  "licence": "MIT-repository; underlying-CDLI-terms", "lines": [], "transliteration": "",
                  "raw_file": str(path), "quality_flags": ["document_identity_unavailable"],
                  "segments": [{"id": str(i+1), "src": s, "tgt": t, "line_ids": [],
                                "alignment": "published_parallel_file", "original_split": path.stem}
                               for i, (s, t) in enumerate(zip(src, tgt)) if s.strip() and t.strip()]}
        ingest([record])
        db.executemany("INSERT OR IGNORE INTO exposure VALUES(?,?)",
                       ((digest(loose_fingerprint(s)), "mtaac_published") for s in src if ws(s)))


def index_photos(raw, db):
    import pyarrow.parquet as pq
    total = 0
    for path in sorted((raw / "photos").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        columns = [n for n in parquet.schema_arrow.names if n != "image"]
        rownum = 0
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                tid = str(row["id"])
                gid = db.execute("SELECT group_id FROM editions WHERE text_id=? LIMIT 1", (tid,)).fetchone()
                iid = f"sumtablets_photos:{path.stem}:{rownum}"
                record = {"id": iid, "tablet_id": tid, "source": "sumtablets_photos",
                          "group_id": gid[0] if gid else None,
                          "kind": "lineart" if "lineart" in path.stem else "photo", "parquet": str(path),
                          "row": rownum, "original_split": path.stem.split("_")[0],
                          "licence": "Apache-2.0-dataset-card; underlying-image-rights-see-CDLI",
                          "alignment": "tablet_composite_views_unverified", "metadata": row}
                db.execute("INSERT OR REPLACE INTO images VALUES(?,?,?,?,?)",
                           (iid, tid, gid[0] if gid else None, None, pack(record)))
                rownum += 1
        total += rownum
        print(f"indexed {path.name}: {rownum:,} images", flush=True)
    return total


def index_maicubeda(raw, db, root):
    from .media import index_maicubeda_archive
    index_maicubeda_archive(raw, db, root)


def jsonl_write(file, record):
    file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def export(root: Path):
    db = connect(root)
    out = root / "exports"
    out.mkdir(exist_ok=True)
    # Точный train неизвестен; весь CDLI исключаем из внешнего теста.
    known_groups = {r[0] for r in db.execute("SELECT DISTINCT group_id FROM editions WHERE source IN ('sumtablets','mtaac_atf','cdli','oracc/cdli')")}
    known_keys = {r[0] for r in db.execute("SELECT key FROM exposure")}
    excluded_groups = {r[0] for r in db.execute("SELECT DISTINCT e.group_id FROM editions e JOIN excluded_documents q ON q.uid=e.uid")}
    known_groups.update(r[0] for r in db.execute("""SELECT DISTINCT e.group_id FROM segments s
        JOIN editions e ON e.uid=s.doc_uid JOIN exposure x ON x.key=s.loose_key
        WHERE s.source!='mtaac_pairs'"""))
    counts = Counter()
    document_info = {}
    for uid, value in db.execute("SELECT uid,data FROM editions WHERE uid IN (SELECT DISTINCT doc_uid FROM segments)"):
        r = unpack(value)
        document_info[uid] = {"url": r.get("url"), "licence": r["licence"],
                              "period": r.get("metadata",{}).get("period"), "genre":r.get("metadata",{}).get("genre")}
    grouped = defaultdict(list)
    for row in db.execute("""SELECT s.id,s.doc_uid,s.source,s.src,s.tgt,s.src_key,s.tgt_key,s.loose_key,s.data,e.group_id
                              FROM segments s JOIN editions e ON s.doc_uid=e.uid ORDER BY s.id"""):
        if row[9] in excluded_groups:
            counts["quality_excluded_segments"] += 1
            continue
        if unpack(row[8]).get("target_lang", "en") != "en":
            counts["non_english_segments_excluded"] += 1
            continue
        grouped[row[7]].append(row)
    files = {}
    for task, splits in [("mt", ("train", "valid", "test", "external_candidates", "ambiguous"))]:
        for split in splits:
            folder = out / task
            folder.mkdir(exist_ok=True)
            files[split] = (folder / f"{split}.jsonl").open("w", encoding="utf-8")
    for key, rows in sorted(grouped.items()):
        # Предпочитаем исходные редакции плоским выгрузкам.
        rows.sort(key=lambda r: (r[2] == "mtaac_pairs", r[0]))
        row = rows[0]
        targets = defaultdict(list)
        for r in rows:
            targets[r[6]].append(r[4])
        groups = sorted({r[9] for r in rows if r[2] != "mtaac_pairs"}) or ["anonymous:" + key[:24]]
        exposure = []
        if key in known_keys:
            exposure.append("published_training_source_text_overlap")
        if any(g in known_groups for g in groups):
            exposure.append("published_training_source_object_overlap")
        if all(r[2] == "mtaac_pairs" for r in rows):
            exposure.append("document_identity_unavailable")
        info = unpack(row[8])
        ambiguous = len(targets) > 1 or bool(info.get("has_variants"))
        src, tgt = row[3], row[4]
        valid = bool(re.search(r"[A-Za-z]", tgt)) and bool(re.search(r"[A-Za-zšĝŋḫŠĜ]", src))
        candidate = not exposure and valid and not ambiguous
        record = {"id": key[:24], "src": src, "tgt": tgt, "group_ids": groups,
                  "sources": sorted({r[2] for r in rows}),
                  "references": sorted({r[4] for r in rows}),
                  "provenance": [{"segment_id": r[0], "document": r[1], "source": r[2],
                                  "url":document_info[r[1]]["url"], "licence":document_info[r[1]]["licence"]} for r in rows],
                  "alignment": info.get("alignment"), "exposure_flags": exposure,
                  "translation_licence": info.get("translation_licence", document_info[row[1]]["licence"]),
                  "periods": sorted({str(document_info[r[1]]["period"]) for r in rows if document_info[r[1]]["period"]}),
                  "genres": sorted({str(document_info[r[1]]["genre"]) for r in rows if document_info[r[1]]["genre"]}),
                  "src_chars": len(src), "tgt_chars":len(tgt),
                  "exposure_status": "candidate_unseen_in_registered_sources" if candidate else "overlap_or_unresolved",
                  "translation_variants": len(targets), "src_hash": key}
        if ambiguous or not valid:
            record["quarantine_reason"] = "alternative_translations_or_source_variants" if ambiguous else "empty_or_unreadable"
            jsonl_write(files["ambiguous"], record)
            counts["mt_ambiguous"] += 1
            continue
        # Одинаковые строки из разных split отправляем в карантин.
        choices = set()
        for g in groups:
            b = int(digest(g)[:8], 16) % 100
            choices.add("train" if g.startswith("anonymous:") else
                        "external_candidates" if g not in known_groups else
                        "test" if b < 10 else "valid" if b < 20 else "train")
        if len(choices) != 1:
            record["quarantine_reason"] = "duplicate_input_across_document_partitions"
            jsonl_write(files["ambiguous"], record)
            counts["mt_ambiguous"] += 1
            continue
        split = choices.pop()
        if split == "external_candidates" and not candidate:
            # Известный текст исключаем даже без привязки к объекту.
            record["quarantine_reason"] = "exposed_text_in_external_document"
            jsonl_write(files["ambiguous"], record)
            counts["mt_ambiguous"] += 1
            continue
        record["split"] = split
        jsonl_write(files[split], record)
        counts["mt_" + split] += 1
    for f in files.values():
        f.close()
    from .quality import screen_mt_near_duplicates
    dump(root / "near-duplicate-stats.json", screen_mt_near_duplicates(out / "mt"))
    with gzip.open(out / "documents.jsonl.gz", "wt", encoding="utf-8") as f:
        for uid, gid, data in db.execute("SELECT uid,group_id,data FROM editions ORDER BY uid"):
            record = unpack(data)
            record["group_id"] = gid
            jsonl_write(f, record)
    with gzip.open(out / "segments.jsonl.gz", "wt", encoding="utf-8") as f:
        for sid, uid, source, src, tgt, key, targetkey, loosekey, value, gid in db.execute("""
            SELECT s.*,e.group_id FROM segments s JOIN editions e ON e.uid=s.doc_uid ORDER BY s.id"""):
            record = unpack(value)
            record.update(id=sid, document=uid, source=source, group_id=gid, src=src, tgt=tgt,
                          target_lang=record.get("target_lang", "en"), src_hash=key, loose_src_hash=loosekey,
                          licence=document_info[uid]["licence"], source_url=document_info[uid]["url"])
            jsonl_write(f, record)
    with (out / "images.jsonl").open("w", encoding="utf-8") as f:
        for _, _, _, _, data in db.execute("SELECT * FROM images ORDER BY id"):
            jsonl_write(f, unpack(data))
    export_translit(root, db)
    from .media import export_e2e
    export_e2e(root, db)
    dump(root / "export-counts.json", dict(counts))
    db.close()


def export_translit(root, db):
    folder = root / "exports/translit"
    folder.mkdir(exist_ok=True)
    files = {s: (folder / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "valid", "test", "external_candidates")}
    known = {g for (g,) in db.execute("SELECT DISTINCT group_id FROM editions WHERE source='sumtablets'")}
    excluded = {r[0] for r in db.execute("SELECT DISTINCT e.group_id FROM editions e JOIN excluded_documents q ON q.uid=e.uid")}
    candidates = {}
    for uid, source, tid, gid, data in db.execute("SELECT * FROM editions WHERE source!='sumtablets' ORDER BY uid"):
        if gid in excluded:
            continue
        r = unpack(data)
        for line in r.get("lines", []):
            glyphs = line.get("glyphs", "")
            text = line.get("text", "")
            if not line.get("is_sumerian") or not glyphs or "<UNK>" in glyphs or not text:
                continue
            # Оставляем одну пару знаков и чтения на семейство.
            b = int(digest(gid)[:8], 16) % 100
            split = ("external_candidates" if gid not in known else
                     "test" if b < 10 else "valid" if b < 20 else "train")
            identity = uid + line["id"] if line["id"] else "\0".join([uid, glyphs, text])
            r = {"id": digest(identity)[:24], "tablet": tid,
                                      "group_id": gid, "src": glyphs, "tgt": text,
                                      "source": source, "document": uid, "line_id": line["id"],
                                      "glyphs_origin": "derived_from_editorial_readings",
                                      "exposure_status": "object_in_sumtablets" if gid in known else "candidate_new_object"}
            key = digest(loose_fingerprint(glyphs))
            targetkey = fingerprint(text)
            previous = candidates.get(key)
            if previous is None:
                candidates[key] = (split, targetkey, r)
            elif previous[0] != split or previous[1] != targetkey:
                # Разные чтения или split для одних знаков неоднозначны.
                candidates[key] = ("ambiguous", "", None)
    for split, _, r in candidates.values():
        if split != "ambiguous":
            jsonl_write(files[split], r)
    for f in files.values():
        f.close()


def audit(root: Path):
    db = connect(root)
    report = {"editions": db.execute("SELECT count(*) FROM editions").fetchone()[0],
              "object_or_work_groups": db.execute("SELECT count(DISTINCT group_id) FROM editions").fetchone()[0],
              "aligned_segments_before_dedup": db.execute("SELECT count(*) FROM segments").fetchone()[0],
              "images": db.execute("SELECT count(*) FROM images").fetchone()[0],
              "sources": dict(db.execute("SELECT source,count(*) FROM editions GROUP BY source")),
              "mt": {}, "translit": {}, "checks": {}}
    for task in ("mt", "translit", "ocr_unicode", "ocr_signs"):
        if task not in report:
            report[task] = {}
        groups = {}
        hashes = {}
        for path in sorted((root / "exports" / task).glob("*.jsonl")):
            if path.stem in {"ambiguous", "near_duplicates"}:
                report[task][path.stem] = sum(1 for _ in path.open())
                continue
            gs, hs, n = set(), set(), 0
            with path.open() as f:
                for line in f:
                    r = json.loads(line)
                    gs.update(r.get("group_ids", [r.get("group_id")]))
                    hs.add(r["sha256"] if task.startswith("ocr") else digest(loose_fingerprint(r["src"])))
                    n += 1
            report[task][path.stem] = n
            groups[path.stem], hashes[path.stem] = gs, hs
        for a in sorted(groups):
            for b in sorted(groups):
                if a >= b:
                    continue
                report["checks"][f"{task}_{a}_{b}_group_overlap"] = len(groups[a] & groups[b])
                report["checks"][f"{task}_{a}_{b}_input_overlap"] = len(hashes[a] & hashes[b])
    report["storage_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    report["limitations"] = [
        "Candidate external data are not proof of absence from model pretraining.",
        "SumTablets/ORACC Unicode glyphs are derived readings, not independent OCR ground truth.",
        "Tablet-level image links do not imply line/surface alignment.",
        "Alternative translations are preserved and excluded from strict MT exports.",
        "Licences differ by source; ETCSL includes noncommercial/share-alike restrictions.",
    ]
    dump(root / "audit.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"sources", "checks"}}, ensure_ascii=False, indent=2), flush=True)
    db.close()
    if any(report["checks"].values()):
        raise ValueError("Evaluation split leakage detected; see audit.json")
    if report["storage_bytes"] > 15_000_000_000:
        raise ValueError("Collection exceeds the user's 15 GB storage budget")
    return report
