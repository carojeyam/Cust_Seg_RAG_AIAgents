import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize ChromaDB client and embedding model
client = chromadb.Client()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global cache for collections
_collections_cache = {}

def _load_collection(collection_name: str, data_file: str):
    """Load or create a collection from a data file"""
    if collection_name in _collections_cache:
        return _collections_cache[collection_name]
    
    try:
        collection = client.get_collection(collection_name)
        _collections_cache[collection_name] = collection
        return collection
    except chromadb.errors.NotFoundError:
        # Create new collection and populate it
        collection = client.create_collection(collection_name)
        
        # Load and ingest data
        if os.path.exists(data_file):
            with open(data_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Better chunking: split on double newlines or every ~500 chars
            raw_chunks = text.split("\n\n")
            chunks = []
            for chunk in raw_chunks:
                if len(chunk.strip()) > 0:
                    if len(chunk) > 500:
                        # further split large chunks
                        for i in range(0, len(chunk), 500):
                            chunks.append(chunk[i:i+500])
                    else:
                        chunks.append(chunk)
            
            for i, chunk in enumerate(chunks):
                embedding = embedder.encode(chunk).tolist()
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[f"{collection_name}_{i}"],
                    metadatas=[{"source": os.path.basename(data_file)}]
                )
        
        _collections_cache[collection_name] = collection
        return collection

# Product search
def product_search(query: str, top_k: int = 5):
    try:
        data_file = os.path.join(os.path.dirname(__file__), "..", "data", "product.txt")
        product_db = _load_collection("products", data_file)
        embedding = embedder.encode(query).tolist()
        result = product_db.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        documents = result["documents"][0] if result["documents"] else []
        # Deduplicate and clean
        unique_docs = []
        for doc in documents:
            if doc.strip() and doc not in unique_docs:
                unique_docs.append(doc)
        return unique_docs[:top_k] if unique_docs else ["No matching products found."]
    except Exception as e:
        return [f"Error searching products: {e}"]

# Marketing search
def marketing_search(query: str, top_k: int = 5):
    try:
        data_file = os.path.join(os.path.dirname(__file__), "..", "data", "cust_seg.txt")
        marketing_db = _load_collection("marketing", data_file)
        embedding = embedder.encode(query).tolist()
        result = marketing_db.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        documents = result["documents"][0] if result["documents"] else []
        # Deduplicate and clean
        unique_docs = []
        for doc in documents:
            if doc.strip() and doc not in unique_docs:
                unique_docs.append(doc)
        return unique_docs[:top_k] if unique_docs else ["No matching segments or marketing strategies found."]
    except Exception as e:
        return [f"Error searching marketing data: {e}"]
