import httpx

def main():
    response = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": "Объясни одним предложением, что такое выручка.",
                }
            ],
            "stream": False,
            "think": False,
        },
        timeout=120.0,
    )

    response.raise_for_status()

    data = response.json()
    print(data["message"]["content"])


if __name__ == "__main__":
    main()