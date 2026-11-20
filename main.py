import argparse

from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

DB_PATH = "./chroma_data"
DOCUMENTS_DIR = "docs"


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about the policy documents in docs/."
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Index documents from the docs directory.",
    )
    args = parser.parse_args()

    load_dotenv()
    rag = RAGPipeline(db_path=DB_PATH)

    if args.ingest:
        rag.ingest(DOCUMENTS_DIR)
        print("Document ingestion completed.")
        return

    question = input("Enter your question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    print("\nAnswer:")
    print(rag.ask(question))


if __name__ == "__main__":
    main()
