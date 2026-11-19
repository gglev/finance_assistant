from pathlib import Path
from finance_assistant.pdf import extract_pages_from_pdf
import json

def main():
    pdf_path = Path("data/raw/COCACOLA_2022_10K.pdf")
    pages = extract_pages_from_pdf(pdf_path)

    output_path = Path("data/processed/COCACOLA_2022_10K_pages.json")
    output_path.parent.mkdir(parents=True, exist_ok = True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(pages, file, indent=2, ensure_ascii=False)

    with output_path.open("r", encoding="utf-8") as file:
        saved_pages = json.load(file)

    print("Pages in JSON: ", len(saved_pages))
    print("Data is similar: ", saved_pages == pages)
    
    print("Saved to", output_path)
    
    # print("Amount of pages extracted: ", len(pages))
    # print("Document: ", pages[0]["document"])
    # print("First page text: ", pages[0]["page_number"])
    # print("Text of the 20th page: ", pages[19]["text"][:2000])

if __name__ == "__main__":
    main()