from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import threading
import zipfile
from urllib.parse import urlencode
from datetime import datetime, timezone
from pathlib import Path

ORACC = "https://oracc.museum.upenn.edu"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


class Downloader:
    """Загрузка с докачкой и сохранением метаданных; готовый кэш не обновляется."""

    def __init__(self, root: Path, max_file_bytes: int = 1024**3):
        self.root = root
        self.max_file_bytes = max_file_bytes
        self.cookie_file = None

    def fetch(self, url: str, relative: str, *, timeout=3600) -> Path:
        path = self.root / relative
        receipt = path.with_name(path.name + ".receipt.json")
        if path.exists() and receipt.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        headers = path.with_name(path.name + ".headers")
        if shutil.disk_usage(self.root).free < 3 * 1024**3:
            raise RuntimeError("Less than 3 GiB free; stopping acquisition")
        cmd = ["curl", "--fail", "--location", "--silent", "--show-error",
               "--connect-timeout", "20", "--max-time", str(timeout),
               "--retry", "2", "--retry-delay", "2", "--max-filesize", str(self.max_file_bytes),
               "--user-agent", "sumbase-research-corpus/0.2", "--dump-header", str(headers),
               "--output", str(part), url]
        if self.cookie_file:
            cmd[1:1] = ["--cookie", str(self.cookie_file)]
        # Небольшие файлы проще скачать заново.
        if part.exists() and part.stat().st_size > 1024**2:
            cmd[1:1] = ["--continue-at", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"curl {result.returncode}: {result.stderr.strip()[-500:]}")
        size = part.stat().st_size
        if not size:
            raise ValueError("Empty response")
        with part.open("rb") as f:
            prefix = f.read(200)
        if prefix.startswith(b"version https://git-lfs") or prefix.strip() == b"404":
            raise ValueError("Not dataset content (LFS pointer or soft 404)")
        if path.suffix == ".zip":
            with zipfile.ZipFile(part) as z:
                if not z.namelist():
                    raise ValueError("Empty ZIP archive")
        if path.suffix == ".json":
            # В JSON ORACC встречаются управляющие символы.
            json.loads(part.read_text(encoding="utf-8"), strict=False)
        part.replace(path)
        dump(receipt, {"url": url, "path": relative, "bytes": size,
                       "sha256": checksum(path),
                       "downloaded_at": datetime.now(timezone.utc).isoformat(),
                       "http_headers": headers.read_text(errors="replace")})
        print(f"downloaded {relative}: {size:,} bytes", flush=True)
        return path


def parallel_fetch(dl: Downloader, jobs: list[tuple[str, str]], workers=4) -> list[dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(dl.fetch, url, path): (url, path) for url, path in jobs}
        for fut in concurrent.futures.as_completed(pending):
            url, path = pending[fut]
            try:
                fut.result()
                results.append({"url": url, "path": path, "status": "downloaded"})
            except Exception as exc:
                results.append({"url": url, "path": path, "status": "failed", "error": str(exc)})
                print(f"FAILED {path}: {exc}", flush=True)
    return sorted(results, key=lambda r: r["path"])


def fetch_text(root: Path, workers=4, oracc_scope="sumerian") -> None:
    dl = Downloader(root)
    index = dl.fetch(ORACC + "/json/", "indexes/oracc.html")
    dl.fetch(ORACC + "/json/projects.json", "indexes/oracc-projects.json")
    archives = sorted(set(re.findall(r'href="/json/([^"/]+\.zip)"', index.read_text())))
    # Учитываем также многоязычные коллекции.
    relevant = {"blms", "ckst", "dcclt", "dccmt", "dsst", "edlex", "eisl", "etcsri",
                "epsd2", "obel", "obmc", "obta", "qcat", "ribo", "riao", "ecut",
                "cams", "caspo", "cmawro", "contrib", "glass", "btto", "cdli", "oimea",
                "nere", "kish", "urap", "tsae", "ccpo", "akklove", "balt"}
    if oracc_scope != "all":
        archives = [a for a in archives if a[:-4].split("-")[0] in relevant]
    jobs = [(f"{ORACC}/json/{a}", f"oracc/{a}") for a in archives]

    # Фиксируем ревизию до формирования URL файлов.
    gh = "https://api.github.com/repos/"
    tree = json.loads(dl.fetch(gh + "cdli-gh/data/git/trees/master", "indexes/cdli-tree.json").read_text())
    sha = tree["sha"]
    for name in ("cdli_cat.csv", "cdliatf_unblocked.atf"):
        jobs.append((f"https://media.githubusercontent.com/media/cdli-gh/data/{sha}/{name}", f"cdli/{name}"))
    jobs.append((f"https://raw.githubusercontent.com/cdli-gh/data/{sha}/README.md", "cdli/README.md"))
    repo = "cdli-gh/Semi-Supervised-NMT-for-Sumerian-English"
    tree = json.loads(dl.fetch(gh + repo + "/git/trees/master?recursive=1", "indexes/mtaac-tree.json").read_text())
    sha = tree["sha"]
    for entry in tree["tree"]:
        p = entry["path"]
        if entry["type"] == "blob" and (p.startswith("dataset/") or p == "LICENSE"):
            jobs.append((f"https://raw.githubusercontent.com/{repo}/{sha}/{p}", "mtaac/" + p))
    repo = "cdli-gh/Semantic-Role-Labeler"
    tree = json.loads(dl.fetch(gh + repo + "/git/trees/master?recursive=1", "indexes/etcsl-tree.json").read_text())
    sha = tree["sha"]
    # Архив заменяет отдельные запросы к страницам ETCSL.
    jobs.append((f"https://codeload.github.com/{repo}/zip/{sha}", "etcsl/mirror.zip"))
    hf = json.loads(dl.fetch("https://huggingface.co/api/datasets/colesimmons/SumTablets", "indexes/sumtablets.json").read_text())
    for entry in hf["siblings"]:
        p = entry["rfilename"]
        if p.endswith((".csv", ".md")):
            jobs.append((f"https://huggingface.co/datasets/colesimmons/SumTablets/resolve/{hf['sha']}/{p}", f"sumtablets/{p}"))
    results = parallel_fetch(dl, jobs, workers)
    dump(root / "acquisition-text.json", {"oracc_scope": oracc_scope, "files": results})
    print(f"Text acquisition: {sum(r['status']=='downloaded' for r in results)}/{len(results)} files", flush=True)


def fetch_photos(root: Path, workers=2, budget_gb=4.0) -> None:
    dl = Downloader(root, max_file_bytes=int(budget_gb * 1024**3))
    hf = json.loads(dl.fetch("https://huggingface.co/api/datasets/colesimmons/SumTablets_Photos",
                            "indexes/sumtablets-photos.json").read_text())
    tree = json.loads(dl.fetch(f"https://huggingface.co/api/datasets/colesimmons/SumTablets_Photos/tree/{hf['sha']}/data",
                              "indexes/sumtablets-photos-files.json").read_text())
    known_sizes = {x["path"]: x["size"] for x in tree}
    # Изображения остаются внутри исходных шардов.
    paths = ["data/test_lineart.parquet", "data/validation_lineart.parquet",
             "data/test_photo.parquet", "data/validation_photo.parquet"]
    total = 0
    results = []
    for p in paths:
        relative = "photos/" + p.split("/")[-1]
        url = f"https://huggingface.co/datasets/colesimmons/SumTablets_Photos/resolve/{hf['sha']}/{p}"
        head = subprocess.run(["curl", "-fILsS", "--max-time", "30", url], capture_output=True, text=True)
        sizes = re.findall(r"(?im)^content-length:\s*(\d+)", head.stdout)
        size = known_sizes.get(p, int(sizes[-1]) if sizes else None)
        if size is None or total + size > budget_gb * 1024**3:
            results.append({"path": relative, "url": url, "status": "budget_skipped", "bytes": size})
            continue
        results.extend(parallel_fetch(dl, [(url, relative)], workers))
        total += size
    dump(root / "acquisition-photos.json", {"budget_gb": budget_gb, "files": results})


def fetch_supplement(root: Path, workers=3) -> None:
    dl = Downloader(root, max_file_bytes=3500 * 1024**2)
    base = "https://heidata.uni-heidelberg.de"
    jobs = []
    for short, doi in [("heicubeda", "IE8CCN"), ("maicubeda", "QSNIQ2")]:
        data = json.loads(dl.fetch(f"{base}/api/datasets/:persistentId/?persistentId=doi:10.11588/data/{doi}",
                                  f"indexes/{short}.json").read_text())
        for item in data["data"]["latestVersion"]["files"]:
            f = item["dataFile"]
            name = f["filename"]
            if item.get("restricted"):
                continue
            if f["filesize"] < 100 * 1024**2 or name == "MaiCuBeDa_Annotations_Mixed.zip":
                jobs.append((f"{base}/api/access/datafile/{f['id']}", f"{short}/{name}"))
    dump(root / "acquisition-supplement.json", {"files": parallel_fetch(dl, jobs, workers)})


def fetch_translations(root: Path, workers=2) -> None:
    """Загрузка английских переводов с проверкой шумерской разметки CDL."""
    dl = Downloader(root)
    candidates = {}
    for path in sorted((root / "oracc").glob("*.zip")):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith("/catalogue.json") or not z.getinfo(name).file_size:
                    continue
                cat = json.loads(z.read(name), strict=False)
                project = cat.get("project", name.rsplit("/", 1)[0])
                for tid, meta in cat.get("members", {}).items():
                    if "en" not in (meta.get("trans") or []):
                        continue
                    cdlname = name.rsplit("/", 1)[0] + f"/corpusjson/{tid}.json"
                    if cdlname not in z.namelist():
                        continue
                    content = z.read(cdlname)
                    if not re.search(rb'"lang"\s*:\s*"sux', content):
                        continue
                    # В каталоге project бывает описанием, в CDL — кодом проекта.
                    own = json.loads(content, strict=False).get("project", project)
                    if not re.fullmatch(r"[a-z0-9_-]+(?:/[a-z0-9_-]+)*", own):
                        own = project
                    key = (own, tid)
                    candidates[key] = (f"{ORACC}/{own}/{tid}/html",
                                       f"translations/{own}/{tid}.html")
    jobs = list(candidates.values())
    dump(root / "translation-plan.json", {"count": len(jobs), "files": jobs})
    print(f"English/Sumerian HTML edition candidates: {len(jobs)}", flush=True)
    results = parallel_fetch(dl, jobs, workers)
    dump(root / "acquisition-translations.json", {"files": results})


def fetch_bdtns(root: Path, workers=2) -> None:
    """Загрузка текстов BDTNS, отсутствующих в исходных корпусах."""
    from lxml import html
    from .bdtns_http import BDTNSPublicSession
    base = "https://bdtns.cesga.es/"
    dl = Downloader(root)
    cookies = root / "bdtns/public-session.cookies"
    cookies.parent.mkdir(parents=True, exist_ok=True)
    params = {"numMuseo": "", "numBDTS": "", "numCDLI": "", "procedencia": "TODOS", "sello": "TODOS",
              "fechaPub": "", "datacion": "", "abreviatura": "", "autor": "", "propietario": "",
              "tipoobjeto": "TODOS", "tipotexto": "TODOS", "lexema_sello": "", "tipoperiodo": "TODOS",
              "tipolenguaje": "Sumerian", "orden": "bdts_uriii_ediciones.abreviatura"}
    query = base + "principal.php?" + urlencode(params)
    result = subprocess.run(["curl", "-fLsS", "--max-time", "90", "--cookie-jar", str(cookies), query],
                            capture_output=True)
    result.check_returncode()
    (root / "bdtns/search.html").write_bytes(result.stdout)
    dl.cookie_file = cookies
    first = dl.fetch(base + "columna1.php?lexema_sello=&lexema=", "bdtns/catalogue/000001.html")
    text = first.read_bytes().decode("utf-8", errors="replace")
    m = re.search(r"out of\s+(\d+)", html.fromstring(text).text_content())
    if not m:
        raise ValueError("BDTNS search did not return catalogue count")
    count = int(m.group(1))
    print(f"BDTNS public Sumerian catalogue: {count:,} records", flush=True)
    # У каждого потока своя PHP-сессия.
    local = threading.local()

    def fetch_independent(url, relative):
        if not hasattr(local, "client"):
            local.client = BDTNSPublicSession(root, query)
        return local.client.catalogue(relative)

    def run_jobs(jobs):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
            pending = {pool.submit(fetch_independent, u, p): (u, p) for u, p in jobs}
            for future in concurrent.futures.as_completed(pending):
                u, p = pending[future]
                try:
                    future.result()
                    results.append({"url":u,"path":p,"status":"downloaded"})
                except Exception as exc:
                    results.append({"url":u,"path":p,"status":"failed","error":str(exc)})
                    print(f"FAILED {p}: {exc}", flush=True)
        return results

    for path in (root / "bdtns/catalogue").glob("*.html"):
        tree = html.fromstring(path.read_bytes().decode("utf-8", errors="replace"))
        actual = re.search(r"Texts from\s+(\d+)", " ".join(tree.text_content().split()))
        if not actual or int(actual.group(1)) != int(path.stem):
            # Повреждённый кэш индекса можно пересоздать.
            path.unlink()
            path.with_name(path.name+".receipt.json").unlink(missing_ok=True)
    jobs = [(base + f"columna1.php?bloque=0&ir_a={start}", f"bdtns/catalogue/{start:06d}.html")
            for start in range(101, count + 1, 100)]
    results = run_jobs(jobs)
    # Наличие записи в каталоге не означает наличие текста.
    known = set(re.findall(r"(?m)^&([PQ]\d{6})\s*=", (root / "cdli/cdliatf_unblocked.atf").read_text()))
    known_bdtns = set()
    for path in (root / "oracc").glob("epsd2-admin*.zip"):
        with zipfile.ZipFile(path) as archive:
            archive_names = set(archive.namelist())
            for name in archive.namelist():
                if name.endswith("/catalogue.json") and archive.getinfo(name).file_size:
                    members = json.loads(archive.read(name), strict=False).get("members", {})
                    prefix = name.rsplit("/",1)[0]
                    for tid, record in members.items():
                        bid = str(record.get("bdtns_id", ""))
                        if bid.isdigit():
                            edition = f"{prefix}/corpusjson/{tid}.json"
                            if edition in archive_names and archive.getinfo(edition).file_size:
                                if re.search(rb'"lang"\s*:\s*"sux', archive.read(edition)):
                                    known_bdtns.add(int(bid))
    import csv
    for path in (root / "sumtablets").glob("*.csv"):
        with path.open() as f:
            known.update(r["id"] for r in csv.DictReader(f))
    candidates = {}
    catalogue_rows = []
    for path in sorted((root / "bdtns/catalogue").glob("*.html")):
        tree = html.fromstring(path.read_bytes().decode("utf-8", errors="replace"))
        for anchor in tree.xpath('//a[contains(@href,"ficha_simple_ventana.php?miReferencia=")]'):
            row = anchor.xpath("ancestor::tr[1]")[0]
            content = " ".join(row.text_content().split())
            refs = row.xpath('.//a[contains(@href,"resultado_bibliografia.php?id_bdts=")]/@href')
            if not refs:
                continue
            bid = re.search(r"id_bdts=(\d+)", refs[0]).group(1)
            ids = sorted(set(re.findall(r"P\d{6}", content)))
            item = {"bdtns_id": bid, "cdli_ids": ids, "catalogue_page": str(path),
                    "catalogue_offset": int(path.stem),
                    "query_record_url": base + anchor.get("href"), "catalogue_text": content}
            catalogue_rows.append(item)
            if int(bid) not in known_bdtns and not any(tid in known for tid in ids):
                candidates[bid] = item
    dump(root / "bdtns/catalogue-index.json", catalogue_rows)
    dump(root / "bdtns/novel-plan.json", {"query": query, "catalogue_count": count,
                                         "novel_candidates": len(candidates), "records": list(candidates.values())})
    print(f"BDTNS candidates without baseline text IDs: {len(candidates):,}", flush=True)
    # miReferencia зависит от текущей страницы сессии.
    # Переход на страницу и чтение карточек выполняются последовательно.
    by_page = {}
    for bid, r in candidates.items():
        by_page.setdefault(r["catalogue_offset"], []).append((bid, r))

    def fetch_page_records(offset, records):
        fetch_independent(base + "columna1.php?lexema_sello=&lexema=", "bdtns/catalogue/000001.html")
        missing = [(bid, r) for bid,r in records if not (root/f"bdtns/records/{bid}.html").exists()]
        if missing:
            local.client.activate(offset)
        output = []
        for bid, r in records:
            relative = f"bdtns/records/{bid}.html"
            try:
                local.client.record(bid)
                output.append({"path":relative,"status":"downloaded","bdtns_id":bid})
            except Exception as exc:
                output.append({"path":relative,"status":"failed","error":str(exc)})
        return output

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
        futures = {pool.submit(fetch_page_records, offset, records): offset for offset, records in sorted(by_page.items())}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:
                results.append({"catalogue_offset": futures[future], "status":"failed", "error":str(exc)})
            if len(results) % 100 < 5:
                print(f"BDTNS progress: {len(results):,} catalogue/record outcomes", flush=True)
    dump(root / "acquisition-bdtns.json", {"files": results})


def fetch_inventory(root: Path, workers=3):
    dl = Downloader(root)
    repos = ["SzuTao/Cuneiform", "Falah/sumerian_prompts", "wheelofheaven/sumerian-king-list-woh",
             "TRACCERR/Sumtables-Cuneiform-Full-Fable5-Remaster"]
    jobs = []
    for repo in repos:
        name = repo.replace("/", "--")
        data = json.loads(dl.fetch("https://huggingface.co/api/datasets/"+repo, f"indexes/extra-{name}.json").read_text())
        for entry in data["siblings"]:
            p = entry["rfilename"]
            if p.endswith(".md") or (repo != repos[-1] and p.endswith((".csv", ".jsonl", ".json", ".parquet"))):
                jobs.append((f"https://huggingface.co/datasets/{repo}/resolve/{data['sha']}/{p}", f"additional/{name}/{p}"))
    for repo in ("praeclarum/cuneiform", "colesimmons/sumerian-transliteration", "colesimmons/lineart-ocr"):
        name = repo.replace("/", "--")
        info = json.loads(dl.fetch("https://huggingface.co/api/models/"+repo, f"indexes/model-{name}.json").read_text())
        jobs.append((f"https://huggingface.co/{repo}/resolve/{info['sha']}/README.md", f"model-cards/{name}.md"))
    jobs += [("https://zenodo.org/api/records/10018951", "indexes/ebl-zenodo.json"),
             ("https://zenodo.org/api/records/10018951/files/fragments.json/content", "ebl-fragments.json"),
             ("https://bdtns.cesga.es/extras/doc/2011_12_20_Copyright_BDTNS.pdf", "bdtns/COPYRIGHT.pdf"),
             ("https://oracc.museum.upenn.edu/doc/opendata/json/index.html", "indexes/oracc-json-documentation.html"),
             ("https://cdli.earth/docs/api", "indexes/cdli-api-documentation.html")]
    dump(root / "acquisition-inventory.json", {"files":parallel_fetch(dl,jobs,workers)})
