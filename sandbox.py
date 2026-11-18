import chromadb
from sentence_transformers import SentenceTransformer

from src.RAGPipeline import EMBEDDING_MODEL_NAME

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 로컬 디스크에 저장
client = chromadb.PersistentClient(path="./chroma_data")

collection = client.get_or_create_collection(
    name="policy_demo_bge_m3_v1",
    embedding_function=None,  # 임베딩은 아래에서 직접 생성
    configuration={"hnsw": {"space": "cosine"}},
)

chunks = [
    {
        "id": "equipment:v2:0001",
        "document": (
            "Full-time employees may claim up to $500 per year "
            "for home-office equipment."
        ),
        "metadata": {
            "doc_id": "equipment",
            "document_name": "equipment_policy_v2.md",
            "section": "Eligibility and limits",
            "version": "v2",
            "chunk_index": 0,
        },
    },
    {
        "id": "pto:v1:0001",
        "document": (
            "Employees must submit vacation requests "
            "at least two weeks in advance."
        ),
        "metadata": {
            "doc_id": "pto",
            "document_name": "pto_policy_v1.md",
            "section": "Request process",
            "version": "v1",
            "chunk_index": 0,
        },
    },
]

texts = [chunk["document"] for chunk in chunks]

collection.upsert(
    ids=[chunk["id"] for chunk in chunks],
    documents=texts,
    metadatas=[chunk["metadata"] for chunk in chunks],
    embeddings=model.encode(
        texts, normalize_embeddings=True
    ).tolist(),
)

question = "Can I get reimbursed for a home-office monitor?"

results = collection.query(
    query_embeddings=model.encode(
        [question], normalize_embeddings=True
    ).tolist(),
    n_results=2,
    include=["documents", "metadatas", "distances"],
)

print(results)
