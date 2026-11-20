from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []

    for page in pages:
        page_chunks = text_splitter.split_text(page["text"])

        for chunk_number, text in enumerate(page_chunks):
            chunks.append({
                "document": page["document"],
                "page_number": page["page_number"],
                "chunk_number": chunk_number + 1,
                "text": text
            })
    return chunks