"""Доступ к BDTNS с сохранением сессии и проверкой ID."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from lxml import html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .acquire import checksum, dump


class BDTNSPublicSession:
    def __init__(self, root: Path, query: str):
        self.root, self.query = root, query
        self.base = "https://bdtns.cesga.es/"
        self.session = None
        self.started = 0
        self.offset = None
        self.references = {}

    def get(self, url):
        response = self.session.get(url, timeout=(15, 45))
        response.raise_for_status()
        if len(response.content) > 2 * 1024**2:
            raise ValueError("Unexpectedly large BDTNS response")
        return response

    def initialize(self):
        if self.session is not None:
            self.session.close()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "sumbase-research-corpus/0.2"
        retry = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.session.mount(
            "https://", HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        )
        self.get(self.query)
        first = self.get(self.base + "columna1.php?lexema_sello=&lexema=")
        self.started = time.monotonic()
        self.offset = None
        return first

    @staticmethod
    def parse_page(content, offset):
        tree = html.fromstring(content.decode("utf-8", errors="replace"))
        text = " ".join(tree.text_content().split())
        if not re.search(rf"Texts from\s+{offset}\s+", text):
            raise ValueError(f"BDTNS offset mismatch: expected {offset}")
        references = {}
        for anchor in tree.xpath('//a[contains(@href,"ficha_simple_ventana.php?miReferencia=")]'):
            row = anchor.xpath("ancestor::tr[1]")[0]
            links = row.xpath('.//a[contains(@href,"resultado_bibliografia.php?id_bdts=")]/@href')
            if links:
                bid = re.search(r"id_bdts=(\d+)", links[0]).group(1)
                references[int(bid)] = anchor.get("href")
        return references

    def activate(self, offset, *, force=False):
        if force or self.session is None or time.monotonic() - self.started > 300:
            self.initialize()
        page = self.get(self.base + f"columna1.php?bloque=0&ir_a={offset}")
        self.references = self.parse_page(page.content, offset)
        self.offset = offset
        return page

    def save(self, response, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(response.content)
        part.replace(path)
        dump(
            path.with_name(path.name + ".receipt.json"),
            {
                "url": response.url,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": checksum(path),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "http_headers": "\n".join(f"{k}: {v}" for k, v in response.headers.items()),
                "anonymous_query": self.query,
                "catalogue_offset": self.offset,
            },
        )
        return path

    def catalogue(self, relative):
        path = self.root / relative
        if path.exists():
            self.parse_page(path.read_bytes(), int(path.stem))
            return path
        return self.save(self.activate(int(path.stem)), relative)

    @staticmethod
    def matches(content, bid):
        text = html.fromstring(content.decode("utf-8", errors="replace")).text_content()
        return bool(re.search(rf"BDTNS\s+0*{int(bid)}\b", text))

    def record(self, bid):
        relative = f"bdtns/records/{bid}.html"
        path = self.root / relative
        if path.exists() and self.matches(path.read_bytes(), bid):
            return path
        for attempt in range(2):
            href = self.references.get(int(bid))
            if href is None:
                raise ValueError(
                    f"BDTNS {bid} absent from current page {self.offset}; catalogue ordering changed"
                )
            response = self.get(self.base + href)
            if self.matches(response.content, bid):
                return self.save(response, relative)
            # После истечения сессии восстанавливаем поиск и текущую страницу.
            if attempt == 0:
                self.activate(self.offset, force=True)
        raise ValueError(f"BDTNS {bid}: source returned no matching record after session refresh")
