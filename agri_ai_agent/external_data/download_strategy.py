import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

FORMAT_ORDER = {
    "parquet": 0,
    "arrow": 1,
    "csv": 2,
    "json": 3,
    "xml": 4,
    "html": 5,
    "pdf": 6,
}

STRUCTURED_FORMATS = frozenset({"parquet", "arrow", "csv", "json"})


def format_priority(fmt: str) -> int:
    return FORMAT_ORDER.get(fmt.lower(), 99)


def is_structured(fmt: str) -> bool:
    return fmt.lower() in STRUCTURED_FORMATS


class FormatFilter:
    @staticmethod
    def best(available: list[str]) -> str | None:
        has = any(is_structured(f) for f in available)
        candidates = [f for f in available if not (f.lower() == "pdf" and has)]
        if not candidates:
            return None
        return min(candidates, key=format_priority)

    @staticmethod
    def sorted_files(files: list[dict]) -> list[dict]:
        has = any(is_structured(f.get("format", "")) for f in files)
        scored = []
        for f in files:
            p = format_priority(f.get("format", ""))
            if f.get("format", "").lower() == "pdf" and has:
                p += 100
            scored.append((p, f))
        scored.sort(key=lambda x: x[0])
        return [f for _, f in scored]


def try_priority_downloads(
    attempts: list[tuple[str, str, Path]],
    timeout: int = 120,
) -> Path | None:
    has_structured = any(is_structured(fmt) for _, fmt, _ in attempts)
    scored = []
    for url, fmt, path in attempts:
        p = format_priority(fmt)
        if fmt.lower() == "pdf" and has_structured:
            p += 100
        scored.append((p, url, fmt, path))
    scored.sort(key=lambda x: x[0])

    for _, url, fmt, path in scored:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            logger.info("Downloaded %s -> %s (%s)", fmt, path.name, url)
            return path
        except Exception as e:
            logger.debug("Download failed for %s: %s", url, e)
            continue
    return None


def download_huggingface_parquet(
    dataset_id: str,
    target_dir: Path,
    cache_dir: Path | None = None,
) -> Path | None:
    import datasets as hf_datasets

    target_dir.mkdir(parents=True, exist_ok=True)
    safe = dataset_id.replace("/", "_").replace("-", "_")

    try:
        ds = hf_datasets.load_dataset(
            dataset_id,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:
        logger.warning("load_dataset(%s) failed: %s", dataset_id, exc)
        return None

    split = "train" if "train" in ds else next(iter(ds))
    df = ds[split].to_pandas()
    parquet_path = target_dir / f"{safe}.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(
        "HF dataset %s (%s split, %d rows) saved to %s",
        dataset_id,
        split,
        len(df),
        parquet_path,
    )
    return parquet_path
