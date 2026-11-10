from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache

from .acquire import dump


def index_maicubeda_archive(raw, db, root):
    from .build import digest, jsonl_write, pack

    path = raw / "maicubeda/MaiCuBeDa_Annotations_Mixed.zip"
    meta = raw / "maicubeda/translitmetadata.csv"
    if not path.exists() or not meta.exists():
        return
    folder = root / "exports/ocr_signs"
    folder.mkdir(parents=True, exist_ok=True)
    unicode_folder = root / "exports/ocr_unicode"
    unicode_folder.mkdir(parents=True, exist_ok=True)
    sign_to_unicode = defaultdict(set)
    excluded = {
        r[0]
        for r in db.execute(
            "SELECT DISTINCT e.group_id FROM editions e JOIN excluded_documents q ON q.uid=e.uid"
        )
    }
    with zipfile.ZipFile(raw / "maicubeda/utils.zip") as utility:
        value = utility.read("utils/unicode_to_signnames.json").decode()
        mapping = json.loads(value[value.index("{") : value.rindex("}") + 1])
        for code, item in mapping.items():
            sign_to_unicode[normalize_sign_name(item["signName"])].add(
                chr(int(code.removeprefix("U+"), 16))
            )
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        stats = Counter()
        records = []
        language_cache = {}
        with meta.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                evidence = "MaiCuBeDa CSV explicit language"
                if not row["Language"]:
                    tid = row["CDLI_Number"]
                    if tid not in language_cache:
                        from .build import unpack

                        language_cache[tid] = any(
                            unpack(value).get("metadata", {}).get("language", "").strip().lower()
                            == "sumerian"
                            for (value,) in db.execute(
                                "SELECT data FROM editions WHERE text_id=?", (tid,)
                            )
                        )
                    if language_cache[tid]:
                        evidence = "explicit Sumerian catalogue metadata linked by CDLI ID"
                        stats["language_resolved_by_catalogue"] += 1
                    else:
                        stats["not_explicit_sumerian"] += 1
                        continue
                elif row["Language"] != "Sumerian":
                    stats["not_explicit_sumerian"] += 1
                    continue
                member = maicubeda_member(row["Filename"])
                if member not in names:
                    stats["missing_image"] += 1
                    continue
                tid = row["CDLI_Number"]
                found = db.execute(
                    "SELECT group_id FROM editions WHERE text_id=? LIMIT 1", (tid,)
                ).fetchone()
                gid = found[0] if found else digest(tid)[:24]
                if gid in excluded:
                    stats["catalogued_modern_fake"] += 1
                    continue
                sha = hashlib.sha256(z.read(member)).hexdigest()
                iid = "maicubeda:" + row["Filename"]
                record = {
                    "id": iid,
                    "tablet_id": tid,
                    "group_id": gid,
                    "source": "maicubeda",
                    "kind": "rendered_sign_crop",
                    "archive": str(path),
                    "member": member,
                    "sha256": sha,
                    "licence": "CC-BY-SA-4.0",
                    "alignment": "annotated_sign_bbox",
                    "sign_name": row["charname"],
                    "language_evidence": evidence,
                    "filename_mapping": "published_csv_extra_index_removed_and_extension_png",
                    "transliteration": row["transliteration"],
                    "iiif_url": row.get("", ""),
                    "metadata": row,
                }
                db.execute(
                    "INSERT OR IGNORE INTO images VALUES(?,?,?,?,?)",
                    (iid, tid, gid, sha, pack(record)),
                )
                records.append(record)
                stats["sumerian_sign_crops"] += 1
        by_hash = defaultdict(list)
        for r in records:
            by_hash[r["sha256"]].append(r)
        files = {
            s: (folder / f"{s}.jsonl").open("w", encoding="utf-8")
            for s in ("train", "valid", "test", "ambiguous")
        }
        ufiles = {
            s: (unicode_folder / f"{s}.jsonl").open("w", encoding="utf-8")
            for s in ("train", "valid", "test")
        }
        for sha, rows in sorted(by_hash.items()):
            r = rows[0]
            choices = {
                (
                    "test"
                    if int(digest(x["group_id"])[:8], 16) % 100 < 10
                    else "valid" if int(digest(x["group_id"])[:8], 16) % 100 < 20 else "train"
                )
                for x in rows
            }
            split = (
                choices.pop()
                if len(choices) == 1 and len({x["sign_name"] for x in rows}) == 1
                else "ambiguous"
            )
            r = dict(
                r,
                tgt=r["sign_name"],
                task="image_to_sign_class",
                split=split,
                image_ref={"archive": str(path), "member": r["member"]},
                annotation_ids=[x["id"] for x in rows],
            )
            jsonl_write(files[split], r)
            stats["export_" + split] += 1
            glyphs = sign_to_unicode.get(normalize_sign_name(r["sign_name"]), set())
            if split != "ambiguous" and len(glyphs) == 1:
                jsonl_write(
                    ufiles[split],
                    dict(
                        r,
                        tgt=next(iter(glyphs)),
                        task="image_to_unicode_sign",
                        glyphs_origin="human_sign_class_mapped_to_unicode",
                    ),
                )
                stats["unicode_" + split] += 1
        for f in files.values():
            f.close()
        for f in ufiles.values():
            f.close()
        stats["archive_members"] = len(names)
        dump(root / "maicubeda-stats.json", dict(stats))
        print(f"MaiCuBeDa: {dict(stats)}", flush=True)


def normalize_sign_name(value):
    return re.sub(r"\s+", "", value.replace("_", " ").upper())


def maicubeda_member(filename):
    # В CSV указан JPG и лишний индекс, в архиве — PNG без этого индекса.
    stem = filename.rsplit(".", 1)[0]
    before, after = stem.split("_HS_", 1)
    parts = before.rsplit("_", 4)
    if len(parts) != 5:
        raise ValueError(f"Unexpected MaiCuBeDa filename: {filename}")
    return "char/" + "_".join([parts[0], *parts[2:]]) + "_HS_" + after + ".png"


@lru_cache(maxsize=4)
def _image_archive(path):
    return zipfile.ZipFile(path)


def load_image(record):
    """Чтение изображения из архива без распаковки на диск."""
    from PIL import Image

    ref = record.get("image_ref", record)
    if "archive" in ref:
        z = _image_archive(ref["archive"])
        return Image.open(io.BytesIO(z.read(ref["member"]))).convert("RGB")
    if "parquet" in ref:
        import pyarrow.parquet as pq

        offset = int(ref["row"])
        pf = pq.ParquetFile(ref["parquet"])
        for i in range(pf.num_row_groups):
            n = pf.metadata.row_group(i).num_rows
            if offset < n:
                value = pf.read_row_group(i, columns=["image"]).column("image")[offset].as_py()
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            offset -= n
        raise IndexError(ref["row"])
    return Image.open(ref.get("image", ref.get("path"))).convert("RGB")


def export_e2e(root, db):
    from .build import digest, jsonl_write, unpack

    folder = root / "exports/e2e"
    folder.mkdir(parents=True, exist_ok=True)
    images = defaultdict(list)
    for tid, data in db.execute(
        "SELECT tablet_id,data FROM images WHERE id LIKE 'sumtablets_photos:%'"
    ):
        images[tid].append(unpack(data))
    n = 0
    with (folder / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for uid, tid, gid, data in db.execute(
            "SELECT uid,text_id,group_id,data FROM editions WHERE source!='mtaac_pairs'"
        ):
            if tid not in images or not re.fullmatch(r"P\d{6}", tid):
                continue
            r = unpack(data)
            if "catalogued_modern_fake" in r.get("quality_flags", []):
                continue
            if "composite" in r.get("title", "").lower():
                continue
            lines = r.get("lines", [])
            if not lines or any(
                not x.get("is_sumerian", x.get("language", "").startswith("sux")) for x in lines
            ):
                continue
            segments = [
                unpack(x[0])
                for x in db.execute(
                    "SELECT data FROM segments WHERE doc_uid=? ORDER BY rowid", (uid,)
                )
            ]
            covered = Counter(lid for s in segments for lid in s.get("line_ids", []))
            ids = [x["id"] for x in lines]
            if not ids or any(covered[x] != 1 for x in ids) or set(covered) != set(ids):
                continue
            if any(s.get("has_variants") for s in segments):
                continue
            if any(s.get("target_lang", "en") != "en" for s in segments):
                continue
            target = "\n".join(s["tgt"] for s in segments)
            for img in images[tid]:
                jsonl_write(
                    f,
                    {
                        "id": digest(uid + img["id"])[:24],
                        "tablet": tid,
                        "group_id": gid,
                        "document": uid,
                        "transliteration": r["transliteration"],
                        "translation": target,
                        "translation_line_coverage": 1.0,
                        "image_ref": {"parquet": img["parquet"], "row": img["row"]},
                        "image_kind": img["kind"],
                        "alignment": "tablet_id_only_needs_visual_review",
                        "evaluation_ready": False,
                        "licences": [r["licence"], img["licence"]],
                    },
                )
                n += 1
    dump(root / "e2e-stats.json", {"candidate_triplets": n, "visually_verified_triplets": 0})
    print(f"E2E candidate triplets with full English line coverage: {n}", flush=True)
