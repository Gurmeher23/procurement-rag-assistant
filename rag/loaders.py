"""Document loading. Supports .txt / .md natively and .pdf when pypdf is
installed. Each document becomes {source, text}."""
from __future__ import annotations

import glob
import os


def load_documents(folder: str) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".txt", ".md"}:
            with open(path, encoding="utf-8") as fh:
                docs.append({"source": os.path.basename(path), "text": fh.read()})
        elif ext == ".pdf":
            docs.append({"source": os.path.basename(path), "text": _read_pdf(path)})
    return docs


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install pypdf to load PDF documents.") from e
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
