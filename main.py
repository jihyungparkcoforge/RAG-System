import argparse

from src.RAGPipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Index documents from the docs directory.",
    )
    args = parser.parse_args()

    rag = RAGPipeline(db_path="./chroma_data")

    if args.ingest:
        rag.ingest("docs")
        print("Document ingestion completed.")
        return

    question = input("Enter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    answer = rag.ask(question)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
