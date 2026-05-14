from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.append(f"\n\n===== Page {page_index} =====\n{text}")
    return "".join(chunks).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in args.pdfs:
        out_path = args.out_dir / f"{pdf_path.stem}.txt"
        out_path.write_text(extract_pdf(pdf_path), encoding="utf-8")
        print(f"{pdf_path.name}: wrote {out_path}")


if __name__ == "__main__":
    main()
