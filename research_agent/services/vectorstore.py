

import chromadb
import ollama
import uuid

# Initialize Chroma client (persistent on disk)
chroma_client = chromadb.PersistentClient(path="./chroma")

# Create or get a collection
# IMPORTANT: embedding dimension is inferred automatically by Chroma,
# so you do NOT need to manually specify it if using "nomic-embed-text".
collection = chroma_client.get_or_create_collection(
    name="research_memory",
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)


def embed(text: str):
    """
    Get vector embedding from Ollama.
    Using nomic-embed-text (recommended).
    """
    resp = ollama.embed(
        model="nomic-embed-text",
        input=text
    )
    return resp["embeddings"][0]


def embed_and_store(text: str, metadata: dict = None):
    """
    Stores text chunk(s) with vector embeddings into chromadb.
    """
    vector = embed(text)
    
    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[vector],
        documents=[text],
        metadatas=[metadata or {}]
    )


def search(query: str, k: int = 5):
    """
    Retrieve relevant documents from the vector store.
    """
    query_embedding = embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    # Normalize response for easier usage
    docs = []
    if results and "documents" in results:
        for i in range(len(results["documents"][0])):
            docs.append({
                "document": results["documents"][0][i],
                "metadata": results.get("metadatas", [{}])[0][i],
                "distance": results["distances"][0][i],
            })

    return docs
