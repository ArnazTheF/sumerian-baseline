from __future__ import annotations

import csv
import html as html_std
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def ws(s) -> str:
    return " ".join(unicodedata.normalize("NFC", str(s or "")).split())


def canonical_work_id(value):
    return "etcsl:" + ".".join(str(int(x)) for x in value.removeprefix("etcsl:").split("."))


def fingerprint(s: str) -> str:
    """Ключ с сохранением чисел, имён и маркеров повреждений."""
    from ..normalize import to_cdli_ascii
    s = to_cdli_ascii(unicodedata.normalize("NFC", s)).lower()
    s = s.replace("geš", "gesz").replace("×", "x")
    s = re.sub(r"<\s*(?:surface|column)\s*>", " ", s, flags=re.I)
    return ws(s)


def loose_fingerprint(s: str) -> str:
    # Только для поиска пересечений, не для объединения чтений.
    return ws(re.sub(r"[\[\]⸢⸣⌈⌉#?!*<>]", "", fingerprint(s)))


def atf_documents(path: Path, source: str, licence: str, metadata=None):
    metadata = metadata or {}
    record = None
    language = ""
    surface = ""
    column = ""
    current = None
    occurrence = Counter()

    def finish():
        if not record:
            return None
        if not record["lines"]:
            return None
        record["transliteration"] = "\n".join(x["text"] for x in record["lines"])
        record["language"] = "sux"
        return record

    with path.open(encoding="utf-8-sig", errors="strict") as f:
        for raw in f:
            line = raw.strip().lstrip("\ufeff")
            header = re.match(r"&([PQX]\d+)\s*=\s*(.*)", line)
            if header:
                previous = finish()
                if previous:
                    yield previous
                tid, title = header.groups()
                occurrence[tid] += 1
                key = tid if occurrence[tid] == 1 else f"{tid}:edition{occurrence[tid]}"
                record = {"uid": f"{source}:{key}", "source": source, "text_id": tid,
                          "aliases": [tid], "title": title, "licence": licence,
                          "metadata": metadata.get(tid, {}), "lines": [], "segments": [],
                          "raw_file": str(path), "glyphs_origin": None,
                          "url": f"https://cdli.earth/artifacts/{int(tid[1:])}" if tid.startswith("P") else None}
                language = ""
                surface = column = ""
                current = None
                continue
            if record is None:
                continue
            if line.startswith("#atf: lang "):
                language = line.split()[2]
            elif line.startswith("#tr.en:") and current is not None:
                text = line.partition(":")[2].strip()
                if text:
                    record["segments"].append({"id": current["id"], "line_ids": [current["id"]],
                                               "src": current["text"], "tgt": text,
                                               "alignment": "atf_inline_line"})
            elif line.startswith("@column"):
                column = line[1:]
                current = None
            elif line.startswith("@"):
                surface = line[1:]
                column = ""
                current = None
            else:
                m = re.match(r"(\S+)\.\s+(.+)", line)
                if m and language.startswith("sux"):
                    label, text = m.groups()
                    # Смену языка внутри строки разбираем по токенам.
                    if re.search(r"%\s*(?:akk|hit|elx|arc|uga)", text):
                        current = None
                        continue
                    current = {"id": f"{surface}/{column}/{label}", "label": label,
                               "surface": surface, "column": column, "text": text, "language": language}
                    record["lines"].append(current)
                elif m:
                    current = None
    last = finish()
    if last:
        yield last


def cdl_nodes(nodes):
    for node in nodes:
        yield node
        if isinstance(node, dict):
            # Обходим вложенность CDL, не деревья графем и аннотаций.
            yield from cdl_nodes(node.get("cdl", []))


def gdl_unicode(value):
    if isinstance(value, list):
        return "".join(gdl_unicode(v) for v in value)
    if not isinstance(value, dict):
        return ""
    if value.get("utf8"):
        return value["utf8"]
    for field in ("seq", "group", "gdl"):
        if field in value:
            return gdl_unicode(value[field])
    return "<UNK>"


def cdl_lines(data):
    lines = []
    current = None
    for node in cdl_nodes(data.get("cdl", [])):
        if node.get("type") == "line-start":
            current = {"id": node.get("ref", ""), "label": node.get("label", node.get("n", "")),
                       "forms": [], "glyph_words": [], "languages": []}
            lines.append(current)
        elif node.get("node") == "l":
            f = node.get("f", {})
            if current is None:
                current = {"id": node.get("ref", "").rsplit(".", 1)[0], "label": "",
                           "forms": [], "glyph_words": [], "languages": []}
                lines.append(current)
            if f.get("form"):
                current["forms"].append(f["form"])
                current["languages"].append(f.get("lang", "und"))
                current["glyph_words"].append(gdl_unicode(f.get("gdl", [])) or "<UNK>")
    result = []
    for line in lines:
        languages = set(line.pop("languages"))
        # Для текстовых задач берём только полностью шумерские строки.
        line["is_sumerian"] = bool(languages) and all(x.startswith("sux") for x in languages)
        line["languages"] = sorted(languages)
        line["text"] = " ".join(line.pop("forms"))
        line["glyphs"] = " ".join(line.pop("glyph_words"))
        if line["text"]:
            result.append(line)
    return result


def oracc_documents(path: Path, seen: set[str], html_root: Path):
    from lxml import html

    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        catalogues = {}
        for name in sorted(names):
            if name.endswith("/catalogue.json") and z.getinfo(name).file_size:
                cat = json.loads(z.read(name), strict=False)
                catalogues[name.rsplit("/", 1)[0]] = cat
        for name in sorted(names):
            if "/corpusjson/" not in name or not name.endswith(".json") or not z.getinfo(name).file_size:
                continue
            project = name.split("/corpusjson/")[0]
            tid = Path(name).stem
            path_uid = f"oracc/{project}:{tid}"
            if path_uid in seen:
                continue
            data = json.loads(z.read(name), strict=False)
            project = data.get("project", project)
            uid = f"oracc/{project}:{tid}"
            if uid in seen:
                continue
            seen.add(uid)
            lines = cdl_lines(data)
            suxlines = [x for x in lines if x["is_sumerian"]]
            if not suxlines:
                continue
            cat = catalogues.get(name.split("/corpusjson/")[0], {})
            meta = cat.get("members", {}).get(tid, {})
            aliases = [tid]
            # Композит и его рукописи входят в одну группу разбиения.
            related = []
            for field in ("cdli_composite_id", "cdli_id_numbers", "id_text", "id_composite"):
                related.extend(re.findall(r"[PQ]\d{6}", str(meta.get(field, ""))))
            etcsl = str(meta.get("etcsl_id", meta.get("etcsl", "")))
            work = re.search(r"\d+\.\d+\.\d+(?:\.\d+)?", etcsl)
            if work:
                related.append(canonical_work_id(work.group()))
            segments = []
            hpath = html_root / project / f"{tid}.html"
            if hpath.exists():
                segments = oracc_html_segments(hpath.read_bytes(), lines)
            yield {"uid": uid, "source": f"oracc/{project}", "text_id": tid,
                   "aliases": aliases, "related": sorted(set(related)),
                   "title": meta.get("designation", ""), "metadata": meta,
                   "licence": data.get("license-url", cat.get("license-url", "unknown")),
                   "language": "sux" if len(suxlines) == len(lines) else "sux-mixed",
                   "lines": lines, "segments": segments,
                   "transliteration": "\n".join(x["text"] for x in suxlines),
                   "glyphs": "\n".join(x["glyphs"] for x in suxlines),
                   "glyphs_origin": "derived_from_editorial_readings",
                   "raw_file": str(path), "raw_member": name,
                   "url": f"{ORACC_URL}/{project}/{tid}",
                   "quality_flags": ["cdl_forms_normalized_damage_not_fully_preserved"]}


ORACC_URL = "https://oracc.museum.upenn.edu"


def oracc_html_segments(content: bytes, lines: list[dict]) -> list[dict]:
    from lxml import html
    tree = html.fromstring(content)
    by_id = {x["id"]: x for x in lines}
    result = []
    # Один rowspan может связывать перевод с несколькими строками.
    for table in tree.xpath('//table[.//td[contains(concat(" ",normalize-space(@class)," ")," xtr ")]]'):
        rows = table.xpath("./tr|./tbody/tr")
        for i, row in enumerate(rows):
            for cell in row.xpath('./td[contains(concat(" ",normalize-space(@class)," ")," xtr ")]'):
                cid = cell.get("id", "")
                if not re.search(r"[-_]en(?:[.\-_]|$)", cid):
                    continue
                try:
                    span = int(cell.get("rowspan", "1"))
                except ValueError:
                    continue
                covered_rows = rows[i:i+span]
                covered = [r.get("id") for r in covered_rows if r.get("id") in by_id]
                # Пропускаем оформление, но не неразобранные текстовые строки.
                unparsed = [r for r in covered_rows if r.get("id") not in by_id and "l" in r.get("class", "").split()]
                if not covered or len(covered_rows) != span or unparsed:
                    continue
                selected = [by_id[x] for x in covered]
                if any(not x["is_sumerian"] for x in selected):
                    continue
                # Сноски и подписи не входят в перевод.
                copy = html.fromstring(html.tostring(cell))
                for junk in copy.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," xtr-label ") or contains(concat(" ",normalize-space(@class)," ")," note ") or self::script]'):
                    junk.drop_tree()
                target = ws(copy.text_content())
                if target:
                    result.append({"id": cid, "line_ids": covered,
                                   "src": "\n".join(x["text"] for x in selected), "tgt": target,
                                   "alignment": "oracc_html_rowspan",
                                   "translation_licence": "ORACC-project-HTML-terms; default-CC-BY-SA; see source project",
                                   "includes_nontext_rows": len(covered) != len(covered_rows)})
    if result:
        from langid.langid import LanguageIdentifier, model
        # Под ключом en бывает немецкий; проверяем язык всей редакции.
        global _LANGUAGE_IDENTIFIER
        if _LANGUAGE_IDENTIFIER is None:
            _LANGUAGE_IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)
            _LANGUAGE_IDENTIFIER.set_languages(["en", "de", "fr", "hu", "it", "es"])
        lang, confidence = _LANGUAGE_IDENTIFIER.classify(" ".join(s["tgt"] for s in result))
        for segment in result:
            segment["target_lang"] = lang
            segment["target_language_confidence"] = float(confidence)
    return result


_LANGUAGE_IDENTIFIER = None


def sumtablets_documents(path: Path):
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tid = row["id"]
            yield {"uid": f"sumtablets:{tid}", "source": "sumtablets", "text_id": tid,
                   "aliases": [tid], "title": "", "metadata": {"period": row.get("period"), "genre": row.get("genre")},
                   "language": "sux", "licence": "CC-BY-4.0", "transliteration": row["transliteration"],
                   "glyphs": row["glyphs"], "glyphs_origin": "derived_from_editorial_readings",
                   "lines": [], "segments": [], "original_split": path.stem,
                   "url": "https://huggingface.co/datasets/colesimmons/SumTablets",
                   "raw_file": str(path)}


def etcsl_xml(raw: bytes):
    """Локальное разрешение TEI-сущностей без сетевых запросов."""
    from lxml import etree
    mapping = {"c": "š", "C": "Š", "g": "ĝ", "G": "Ĝ", "h": "ḫ", "H": "Ḫ", "s": "ṣ", "S": "Ṣ", "t": "ṭ", "T": "Ṭ",
               "j": "ĝ", "J": "Ĝ", "X": "...", "commat": "@", "aleph": "ʾ",
               "damb": "⸢", "dame": "⸣", "suppb": "[", "suppe": "]", "qryb": "(?)", "qrye": "", "subb": "", "sube": "", "hr": " — "}
    mapping.update({"s"+str(i): str(i) for i in range(10)})
    determinants = set("ance d dug f gi jic id2 e2 gud iku itid ki im kac ku6 kur kuc lu2 m mu mul mucen na4 ninda sa sar cah2 tug2 tum9 u2 udu urud uzu zabar gesz gisz kusz tug uruda munus".split())
    mapping.update({k: "{" + k.translate(str.maketrans({"c": "š", "j": "ĝ"})) + "}" for k in determinants})
    unknown = set()

    def entity(m):
        key = m.group(1)
        if key in {"amp", "lt", "gt", "quot", "apos"} or key.startswith("#"):
            return m.group(0)
        if key in mapping:
            value = mapping[key]
        elif key + ";" in html_std.entities.html5:
            value = html_std.entities.html5[key + ";"]
        else:
            # Неизвестные сущности сохраняем и отмечаем в отчёте.
            value = "{" + key + "}"
            unknown.add(key)
        return html_std.escape(value, quote=True)

    text = raw.decode("utf-8-sig")
    text = re.sub(r"&([^;\s]+);", entity, text)
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, recover=False)
    return etree.fromstring(text.encode(), parser), unknown


def etcsl_documents(path: Path):
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for name in sorted(names):
            if "/data/etcsl/2518/etcsl/transliterations/c." not in name or not name.endswith(".xml"):
                continue
            tree, unknown = etcsl_xml(z.read(name))
            work = Path(name).name[2:-4]
            tname = name.replace("/transliterations/c.", "/translations/t.")
            ttree, tunknown = etcsl_xml(z.read(tname)) if tname in names else (None, set())
            lines = []
            grouped = defaultdict(list)
            spans = set()
            variants = {}
            # Варианты ETCSL могут задаваться границами span вне дерева XML.
            for elem in tree.xpath(".//text/body//*"):
                if elem.tag in {"addSpan", "secondarySpan"}:
                    spans.add(elem.get("to", ""))
                elif elem.tag == "anchor":
                    spans.discard(elem.get("id", ""))
                elif elem.tag == "l":
                    variants[elem.get("id", "")] = sorted(spans)
            for line in tree.xpath(".//text/body//l"):
                # c и j нормализуем только в шумерском тексте.
                forms = []
                for element in line.iter():
                    if element.tag == "w":
                        forms.append(element.get("form", "").translate(str.maketrans({"c":"š", "C":"Š", "j":"ĝ", "J":"Ĝ"})))
                    elif element.tag == "gap" and not element.xpath("ancestor::w"):
                        forms.append("[...]")
                text = " ".join(f for f in forms if f)
                if not text:
                    continue
                row = {"id": line.get("id", ""), "label": line.get("n", ""), "text": text,
                       "language": "sux", "is_sumerian": True,
                       "variant_context": variants.get(line.get("id", ""), []) + [e.tag + ":" + str(e.get("type", "")) for e in line.iterancestors() if e.tag not in {"body", "text", "TEI.2"}]}
                lines.append(row)
                for ref in line.get("corresp", "").split():
                    grouped[ref.lstrip("#")].append(row)
            segments = []
            if ttree is not None:
                for para in ttree.xpath(".//text/body//p[@id]"):
                    pid = para.get("id")
                    selected = grouped.get(pid, [])
                    if not selected:
                        continue
                    from lxml import etree
                    para_copy = etree.fromstring(etree.tostring(para))
                    for gap in para_copy.xpath(".//gap"):
                        gap.text = " [...] "
                    target = ws("".join(para_copy.itertext()))
                    if target:
                        segments.append({"id": pid, "line_ids": [x["id"] for x in selected],
                                         "src": "\n".join(x["text"] for x in selected), "tgt": target,
                                         "alignment": "etcsl_explicit_corresp",
                                         "has_variants": any(x["variant_context"] for x in selected)})
            yield {"uid": f"etcsl:{work}", "source": "etcsl", "text_id": f"etcsl:{work}",
                   "aliases": [f"etcsl:{work}"], "title": ws("".join(tree.xpath(".//titleStmt/title")[0].itertext())),
                   "metadata": {"genre": "Literary", "witness_references": [ws("".join(x.itertext())) for x in tree.xpath('.//listBibl[@type="tablets"]/bibl')]},
                   "language": "sux", "licence": "CC-BY-NC-SA-3.0", "lines": lines, "segments": segments,
                   "transliteration": "\n".join(x["text"] for x in lines), "glyphs_origin": None,
                   "raw_file": str(path), "raw_member": name,
                   "url": f"https://etcsl.orinst.ox.ac.uk/cgi-bin/etcsl.cgi?text=c.{work}",
                   "quality_flags": ["composite_no_direct_image_alignment"] + (["unresolved_entities:" + ",".join(sorted(unknown | tunknown))] if unknown | tunknown else [])}


def ebl_documents(path: Path):
    for row in json.loads(path.read_text()):
        lines = []
        for n, raw in enumerate(row.get("atf", "").splitlines()):
            m = re.match(r"([^\s]+)\.\s+(.*)", raw)
            if not m:
                continue
            label, content = m.groups()
            # %es — шумерский Emesal, %sb — аккадский. Нужны явные метки.
            for k, span in enumerate(re.finditer(r"%(sux|es)\s+(.+?)(?=\s+%[a-z]|$)", content)):
                text = span.group(2).rstrip(" &")
                if not text:
                    continue
                lines.append({"id": f"{n}:{k}", "label": label, "text": text,
                              "language": "sux-x-emesal" if span.group(1) == "es" else "sux", "is_sumerian": True})
        if not lines:
            continue
        tid = row.get("externalNumbers", {}).get("cdliNumber", "")
        yield {"uid": "ebl:"+row["_id"], "source": "ebl", "text_id": tid or "ebl:"+row["_id"],
               "aliases": [tid] if re.fullmatch(r"P\d{6}", tid) else [], "title": row.get("description", ""),
               "metadata": {k: row.get(k) for k in ("museumNumber", "museum", "collection", "externalNumbers", "record", "script", "genres", "joins")},
               "language": "sux-mixed", "licence": "CC-BY-4.0", "lines": lines, "segments": [],
               "transliteration": "\n".join(x["text"] for x in lines), "raw_file": str(path),
               "url": "https://www.ebl.lmu.de/fragmentarium/"+row["_id"],
               "quality_flags": ["only_explicit_sumerian_language_spans", "unmarked_lines_excluded"]}


def bdtns_documents(folder: Path):
    from lxml import html
    plan = folder / "novel-plan.json"
    if not plan.exists():
        return
    index = {r["bdtns_id"]: r for r in json.loads(plan.read_text())["records"]}
    determinants = set("d ki giš gesz kuš kusz na4 id2 munus lu2 mul mušen uruda tug2 u2 dug ninda".split())
    for path in sorted((folder / "records").glob("*.html")):
        bid = path.stem
        info = index[bid]
        tree = html.fromstring(path.read_bytes().decode("utf-8", errors="replace"))
        for elem in tree.xpath("//script|//style"):
            elem.drop_tree()
        whole = ws(tree.text_content())
        found = re.search(r"BDTNS\s+(\d+)", whole)
        if not found or int(found.group(1)) != int(bid):
            raise ValueError(f"BDTNS session order changed for {path}")
        metadata = {}
        for tr in tree.xpath("//tr"):
            labels = [ws(td.text_content()) for td in tr.xpath("./td")]
            if not labels or labels[0] not in {"Period", "Date", "Collection"}:
                continue
            following_rows = tr.xpath("following-sibling::tr[1]")
            following = following_rows[0] if following_rows else None
            if following is not None:
                values = [ws(td.text_content()) for td in following.xpath("./td")]
                if len(values) == len(labels):
                    metadata.update({k.lower().replace(" ", "_").rstrip("."): v for k,v in zip(labels,values) if v})
        if metadata.get("language") and "sumerian" not in metadata["language"].lower():
            continue
        lines = []
        for tr in tree.xpath("//tr"):
            cells = tr.xpath("./td")
            if len(cells) != 2 or cells[0].get("width") != "10%" or cells[1].get("width") != "90%":
                continue
            label = ws(cells[0].text_content())
            # r. 1 обозначает и оборот, и первую строку на нём.
            if not re.search(r"\d", label):
                continue
            cell = html.fromstring(html.tostring(cells[1]))
            for node in cell.xpath(".//em"):
                node.drop_tree()
            for node in cell.xpath(".//sup"):
                value = ws(node.text_content())
                node.text = "{"+value+"}" if value in determinants else value
                for child in list(node):
                    node.remove(child)
            text = ws(cell.text_content())
            if text:
                lines.append({"id": str(len(lines)), "label": label, "text": text,
                              "language": "sux", "is_sumerian": True})
        if not lines:
            continue
        ids = set(info["cdli_ids"])
        ids.update(re.findall(r"\bCDLI\s+(P\d{6})\b", whole))
        from urllib.parse import urlparse
        for href in tree.xpath("//a/@href"):
            parsed = urlparse(href)
            if parsed.hostname in {"cdli.earth", "cdli.ucla.edu", "cdli.mpiwg-berlin.mpg.de"}:
                ids.update(re.findall(r"P\d{6}", href))
                artifact = re.search(r"/artifacts/(\d+)", parsed.path)
                if artifact:
                    ids.add(f"P{int(artifact.group(1)):06d}")
        ids = sorted(ids)
        yield {"uid": "bdtns:"+bid, "source": "bdtns", "text_id": ids[0] if len(ids) == 1 else "bdtns:"+bid,
               "aliases": ids + ["bdtns:"+bid], "title": info["catalogue_text"][:250],
               "metadata": {**metadata, "bdtns_id": bid,
                            "catalogue_record": info, "record_contributors_and_apparatus": whole[:12000]},
               "language": "sux", "licence": "BDTNS-not-for-profit-research-and-teaching; credit-Manuel-Molina-and-editors",
               "transliteration": "\n".join(x["text"] for x in lines), "lines": lines, "segments": [],
               "raw_file": str(path), "url": info["query_record_url"],
               "quality_flags": ["anonymous_session_url_use_stable_bdtns_id_for_lookup", "html_editorial_emphasis_notes_removed"]}
