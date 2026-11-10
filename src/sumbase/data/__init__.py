from ..config import Config


def prepare(cfg: Config) -> dict:
    if cfg.data.get("source") == "collection":
        from .prepare_collection import prepare_collection

        return prepare_collection(cfg)
    if cfg.task == "mt":
        from .prepare_mt import prepare_mt

        return prepare_mt(cfg)
    if cfg.task == "translit":
        from .prepare_translit import prepare_translit

        return prepare_translit(cfg)
    from .prepare_ocr import prepare_ocr

    return prepare_ocr(cfg)
