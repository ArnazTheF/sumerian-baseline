from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from .acquire import checksum, dump


def verify(root):
    raw = root / "raw"
    failures = []
    checked = 0
    total = 0
    for receipt in sorted(raw.rglob("*.receipt.json")):
        data = json.loads(receipt.read_text())
        path = receipt.with_name(receipt.name.removesuffix(".receipt.json"))
        if not path.exists():
            failures.append({"path": str(path), "error": "missing"})
            continue
        if path.stat().st_size != data["bytes"] or checksum(path) != data["sha256"]:
            failures.append({"path": str(path), "error": "checksum_or_size_mismatch"})
        checked += 1
        total += path.stat().st_size
    result = {"files_checked": checked, "bytes_checked": total, "failures": failures}
    dump(root / "integrity.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise ValueError("Acquired source integrity verification failed")
    return result


def report(root):
    from .build import connect, unpack

    audit = json.loads((root / "audit.json").read_text())
    # Адаптеры могут создавать файлы после аудита сборки.
    audit["storage_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    dump(root / "audit.json", audit)
    db = connect(root)
    lines = Counter()
    periods, genres = Counter(), Counter()
    seen = set()
    unknown_entities = Counter()
    for source, gid, value in db.execute("SELECT source,group_id,data FROM editions ORDER BY uid"):
        r = unpack(value)
        lines[source] += len(r.get("lines", []))
        for flag in r.get("quality_flags", []):
            if flag.startswith("unresolved_entities:"):
                unknown_entities.update(flag.partition(":")[2].split(","))
        if gid not in seen:
            seen.add(gid)
            meta = r.get("metadata", {})
            periods[meta.get("period", "unknown")] += 1
            genres[meta.get("genre", "unknown")] += 1
    extra = {
        "lines_by_source": dict(lines),
        "total_lines_in_editions": sum(lines.values()),
        "periods_one_representative_per_group": dict(periods),
        "genres_one_representative_per_group": dict(genres),
        "unresolved_etcsl_entities": dict(unknown_entities),
    }
    dump(root / "diversity.json", extra)
    locked = []
    for receipt in sorted((root / "raw").rglob("*.receipt.json")):
        item = json.loads(receipt.read_text())
        if not item["path"].startswith(("translations/", "bdtns/catalogue/", "bdtns/records/")):
            locked.append({k: item[k] for k in ("url", "path", "bytes", "sha256", "downloaded_at")})
    acquisition = {}
    for manifest in sorted((root / "raw").glob("acquisition-*.json")):
        values = json.loads(manifest.read_text()).get("files", [])
        acquisition[manifest.stem] = dict(Counter(item["status"] for item in values))
    snapshot = {
        "schema_version": 1,
        "acquisition_started": "2026-11-03",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": locked,
        "acquisition_outcomes": acquisition,
        "oracc_archives_downloaded": len(list((root / "raw/oracc").glob("*.zip"))),
        "oracc_html_editions_downloaded": len(list((root / "raw/translations").rglob("*.html"))),
        "bdtns_catalogue_pages": len(list((root / "raw/bdtns/catalogue").glob("*.html"))),
        "bdtns_record_pages": len(list((root / "raw/bdtns/records").glob("*.html"))),
        "audit": audit,
    }
    for path in [
        root / "raw/bdtns/catalogue-index.json",
        root / "raw/bdtns/novel-plan.json",
        root / "raw/acquisition-translations.json",
    ]:
        if path.exists():
            snapshot.setdefault("index_hashes", {})[str(path.relative_to(root))] = checksum(path)
    dump(root / "sources.lock.json", snapshot)
    db.close()
    text = [
        "# Шумерский корпус: снимок сборки",
        "",
        "Сборка начата 3 ноября 2026 года. Числа ниже получены из локальных файлов, а не из карточек источников.",
        "",
        "## Что собрано",
        "",
        f"- Редакций текстов: **{audit['editions']:,}**. Один объект может иметь несколько редакций.",
        f"- Групп объектов/произведений: **{audit['object_or_work_groups']:,}**. Это технические группы связанных идентификаторов, не число физических табличек.",
        f"- Строк в редакциях, до объединения дублирующих источников: **{sum(lines.values()):,}**.",
        f"- Выровненных переводных сегментов до дедупликации: **{audit['aligned_segments_before_dedup']:,}**.",
        f"- Проиндексированных локальных изображений: **{audit['images']:,}**.",
        f"- Размер каталога: **{audit['storage_bytes']/1e9:.2f} ГБ** ({audit['storage_bytes']/1024**3:.2f} GiB). Лимит пользователя — 15 ГБ.",
        "",
        "## Готовые выборки",
        "",
        "| Задача | train | valid | test | external_candidates | карантин |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in ("mt", "translit", "ocr_unicode", "ocr_signs"):
        counts = audit.get(task, {})
        if counts:
            text.append(
                "| "
                + task
                + " | "
                + " | ".join(
                    str(counts.get(k, 0))
                    for k in ("train", "valid", "test", "external_candidates", "ambiguous")
                )
                + " |"
            )
    text += [
        "",
        "`external_candidates` — кандидаты, не пересекающиеся с зарегистрированными исходными корпусами по доступным идентификаторам и нормализованным текстам. Это **не доказательство**, что готовая модель никогда их не видела. Обычные `valid/test` также не гарантируют независимость от предобучения готовых моделей.",
        "",
        "## Источники",
        "",
        "| Источник редакций | Редакций | Строк до объединения |",
        "|---|---:|---:|",
    ]
    for source, count in sorted(audit["sources"].items()):
        text.append(f"| {source} | {count:,} | {lines[source]:,} |")
    text += [
        "",
        "## Обработка и ограничения",
        "",
        "1. Сохранены исходные файлы, URL, снимки индексов, SHA-256, редакции и происхождение каждого переводного сегмента. Git/Hugging Face URL закреплены ревизиями; для изменяемых архивов ORACC сохранены байты и хеши.",
        "2. Группы соединяют P/Q-идентификаторы, явно указанные отношения композиции и свидетелей, ссылки ETCSL и однозначные музейные номера. Неоднозначные музейные ссылки не соединяются автоматически.",
        "3. Нормализация для поиска совпадений отделена от исходного текста. Сохранены числа, имена и редакторские чтения. Расхождения переводов и исходных вариантов не разрешаются автоматически: они остаются в мастер-корпусе и карантине.",
        "4. Для MT используются явные ATF #tr.en, связи ETCSL corresp и диапазоны строк ORACC HTML. Словарные глоссы не выдаются за перевод. Шумерские и аккадские части двуязычных текстов различаются по языковым тегам. Язык переводов ORACC дополнительно проверяется langid по всей редакции; он может ошибаться на коротких текстах.",
        "5. Строгие разбиения не пересекаются по группам объектов/произведений и нормализованным входам. Для MT длинные близкие к train фрагменты дополнительно вынесены в near_duplicates.jsonl по эвристике триграмм. Полнота поиска парафраз и вариантов написания не гарантируется.",
        "6. Unicode ORACC/SumTablets восстановлен из редакторских чтений. Эти пары полезны для транслитерации, но не являются независимой посимвольной разметкой фотографий. CDL сохраняет не все обозначения повреждений; оригинальные ATF/XML/HTML остаются доступны.",
        "7. В MaiCuBeDa сохранены реальные размеченные вырезки знаков. ocr_signs использует классы знаков; ocr_unicode — только однозначное точное соответствие класса Unicode из опубликованного авторами словаря. Совпадающие байты изображений не пересекают разбиения. Несовпавшие с архивом имена CSV исключены, а не подобраны приблизительно.",
        "8. e2e/candidates.jsonl содержит связки по номеру физической таблички с полным покрытием текстовых строк переводом. Это кандидаты для визуальной проверки: наличие всех поверхностей, качество изображения и соответствие редакции требуют проверки. Текст композиции не присваивается фотографии отдельного свидетеля.",
        "9. Лицензии сохраняются раздельно. ORACC JSON содержит собственные сведения о лицензии; ETCSL — CC BY-NC-SA 3.0; SumTablets — CC BY 4.0; MaiCuBeDa — CC BY-SA 4.0. Карточка SumTablets_Photos указывает Apache-2.0, права на исходные музейные изображения нужно учитывать отдельно. BDTNS разрешает некоммерческое исследовательское/учебное использование при указании правообладателей; это не единая свободная лицензия всего объединения.",
        "",
        "## Аудит",
        "",
        "Машиночитаемые результаты: audit.json, integrity.json, parser-errors.json, near-duplicate-stats.json, diversity.json и acquisition-*.json в каталоге сборки.",
        "",
        "Ошибки или недоступные выгрузки фиксируются в этих файлах. Полнота ограничена доступными публичными версиями и бюджетом хранения; утверждать, что собраны все существующие коллекции мира, нельзя.",
        "",
    ]
    path = root / "REPORT.md"
    path.write_text("\n".join(text), encoding="utf-8")
    print(path)
    return audit
