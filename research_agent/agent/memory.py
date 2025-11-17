
import uuid
from services.vectorstore import embed, collection


class Memory:
    """
    Memory module with vector retrieval using ChromaDB.
    Stores each interaction as a text document and embeds it.
    """

    def save_interaction(self, query: str, answer: str):
        text = f"QUERY:\n{query}\n\nANSWER:\n{answer}"
        vector = embed(text)

        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            documents=[text],
            metadatas=[{"type": "interaction"}]
        )

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieve the top-k most relevant past interactions.
        Returns a list of text snippets.
        """
        query_embedding = embed(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        if "documents" not in results or not results["documents"]:
            return []

        # Flatten
        return results["documents"][0]
