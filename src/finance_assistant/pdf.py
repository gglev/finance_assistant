from pathlib import Path
import pymupdf

pdf_path = Path("data/raw/COCACOLA_2022_10K.pdf")

with pymupdf.open(pdf_path) as doc:
    print("Pages: ", len(doc))
    print(doc[0].get_text()[:1000])

def extract_pages_from_pdf(pdf_path: Path) -> list[dict]:
    pages = []

    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            pages.append({"document": pdf_path.name, "page_number": page_number, "text": page.get_text()})

    return pages