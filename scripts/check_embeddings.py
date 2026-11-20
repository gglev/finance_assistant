from sentence_transformers import SentenceTransformer

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = ["What was the main reason for the company's revenue increase in Q4?",
              "What were the main factors affecting the company's expenses in Q4?"
    ]

    embeddings = model.encode(texts, normalize_embeddings=True)

    print("Length of embeddings: ", len(embeddings))
    print("First five embedding: ", embeddings[0][:5])

if __name__ == "__main__":
    main()