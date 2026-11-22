from pathlib import Path
from finance_assistant.pdf import extract_pages_from_pdf
import json
from finance_assistant.chunking import chunk_pages

def main():
    pdf_path = Path("data/raw/COCACOLA_2022_10K.pdf")
    pages = extract_pages_from_pdf(pdf_path)

    output_path = Path("data/processed/COCACOLA_2022_10K_pages.json")
    output_path.parent.mkdir(parents=True, exist_ok = True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(pages, file, indent=2, ensure_ascii=False)

    with output_path.open("r", encoding="utf-8") as file:
        saved_pages = json.load(file)

    chunks = chunk_pages(saved_pages)
    chunks_path = Path("data/processed/COCACOLA_2022_10K_chunks.json")

    with chunks_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print("Chunks saved to:", chunks_path)
    print("Amount of chunks created: ", len(chunks))
    print("First chunk text: ", chunks[0])
    print("Max length: ", max(len(chunk["text"]) for chunk in chunks))

    print("Pages in JSON: ", len(saved_pages))
    print("Data is similar: ", saved_pages == pages)
    
    print("Saved to", output_path)
    
    # print("Amount of pages extracted: ", len(pages))
    # print("Document: ", pages[0]["document"])
    # print("First page text: ", pages[0]["page_number"])
    # print("Text of the 20th page: ", pages[19]["text"][:2000])

if __name__ == "__main__":
    main()