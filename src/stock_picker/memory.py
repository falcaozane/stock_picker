import chromadb
from sentence_transformers import SentenceTransformer
import uuid

class ChromaMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")

        self.collection = self.client.get_or_create_collection(
            name="stock_picker_memory"
        )

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def save_memory(self, text: str):
        embedding = self.embedding_model.encode(text).tolist()

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())]
        )

    def search_memory(self, query: str, n_results: int = 5):
        embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

        return results["documents"][0]