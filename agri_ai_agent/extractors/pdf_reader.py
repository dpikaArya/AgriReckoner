"""Read a directory of PDFs into {paper_id: text} for LLM extraction."""

import logging
from pathlib import Path

log = logging.getLogger("pdf_reader")


def read_pdf_text(path) -> str:
    """Return the concatenated text of every page of a PDF."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def read_papers_dir(papers_dir) -> dict:
    """Read every ``*.pdf`` in a directory into {stem: text}; unreadable files are skipped."""
    papers = {}
    for pdf_path in sorted(Path(papers_dir).glob("*.pdf")):
        try:
            text = read_pdf_text(pdf_path)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not abort the batch
            log.warning("Skipping unreadable PDF %s: %s", pdf_path.name, exc)
            continue
        if text.strip():
            papers[pdf_path.stem] = text
        else:
            log.warning("No extractable text in %s (scanned image?)", pdf_path.name)
    return papers


if __name__ == "__main__":
    import tempfile

    empty_dir = tempfile.mkdtemp()
    assert read_papers_dir(empty_dir) == {}
    print("pdf_reader smoke OK -> empty dir yields {}")
