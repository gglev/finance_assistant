from pathlib import Path
from finance_assistant.pdf import extract_pages_from_pdf

def main():
    pdf_path = Path("data/raw/COCACOLA_2022_10K.pdf")
    pages = extract_pages_from_pdf(pdf_path)
    
    print("Amount of pages extracted: ", len(pages))
    print("Document: ", pages[0]["document"])
    print("First page text: ", pages[0]["page_number"])
    print("Text of the 20th page: ", pages[19]["text"][:2000])

if __name__ == "__main__":
    main()