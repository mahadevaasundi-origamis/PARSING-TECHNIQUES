import chromadb
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings
import hashlib

# --- Configuration (Assuming these variables are defined upstream) ---
CHROMA_PERSIST_DIRECTORY = "d:/GitHub/Learning/PARSING-TECHNIQUES/Rags/chroma_db"
COLLECTION_NAME = "document_chunks_collection"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


class ChromaDBManager:
    """
    A manager class for ChromaDB to handle creating, searching, and managing a vector collection.
    It uses Ollama for embeddings to stay consistent with the project's stack.
    """
    def __init__(self, persist_directory: str, collection_name: str, embedding_model_name: str):
        """
        Initializes the ChromaDB client and the specific collection.

        Args:
            persist_directory (str): The file path where the database will be stored.
            collection_name (str): The name of the collection to manage.
            embedding_model_name (str): The name of the Ollama model to use for embeddings.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Initialize a persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize the embedding function
        self.embedding_function = OllamaEmbeddings(model=self.embedding_model_name)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            # Note: LangChain's Chroma wrapper handles this, but with a direct client,
            # we don't set the function here. It's used when adding/querying.
        )
        print(f"✅ ChromaDBManager initialized for collection '{self.collection_name}' at '{self.persist_directory}'")

    def _generate_deterministic_id(self, content: str) -> str:
        """
        Generates a deterministic SHA-256 hash for a document's content to use as a unique ID.
        This is a common practice to ensure idempotency (preventing duplicate entries).
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def add_documents(self, documents: List[Document]):
        """
        Adds a list of LangChain Document objects to the collection.
        Embeddings are generated automatically by ChromaDB via the embedding function.
        """
        if not documents:
            print("⚠️ No documents provided to add.")
            return

        contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # Generate unique, deterministic IDs to prevent duplicates
        ids = [self._generate_deterministic_id(content) for content in contents]
        
        # Generate embeddings for the documents
        embeddings = self.embedding_function.embed_documents(contents)

        print(f"📦 Adding {len(documents)} documents to the collection...")
        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Successfully added {len(documents)} documents.")

    def search(self, query_text: str, n_results: int = 5) -> Optional[Dict[str, Any]]:
        """
        Performs a similarity search in the collection.
        The underlying algorithm is typically HNSW, a highly efficient graph-based search.
        """
        if not query_text:
            print("⚠️ Query text cannot be empty.")
            return None
            
        print(f"\n🔍 Searching for '{query_text}'...")
        # Generate embedding for the query
        query_embedding = self.embedding_function.embed_query(query_text)
        
        # Perform the query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single chunk by its unique ID."""
        print(f"📄 Getting chunk with ID: {chunk_id}")
        result = self.collection.get(ids=[chunk_id], include=["metadatas", "documents"])
        if result and result['ids']:
            return {
                "id": result['ids'][0],
                "document": result['documents'][0],
                "metadata": result['metadatas'][0]
            }
        print(f"❌ Chunk with ID '{chunk_id}' not found.")
        return None

    def delete_chunk_by_id(self, chunk_id: str):
        """Deletes a single chunk by its unique ID."""
        print(f"🗑️ Deleting chunk with ID: {chunk_id}")
        self.collection.delete(ids=[chunk_id])
        print(f"✅ Chunk with ID '{chunk_id}' deleted (if it existed).")

    def get_all_chunks(self, limit: int = 100) -> Dict[str, Any]:
        """
        Retrieves all chunks from the collection, up to a specified limit.
        Warning: This can be slow and memory-intensive for very large collections.
        """
        print(f"📚 Retrieving all chunks (limit: {limit})...")
        return self.collection.get(limit=limit, include=["metadatas", "documents"])

    def clear_collection(self):
        """
        Deletes all items within the collection, but keeps the collection itself.
        This is faster than deleting and recreating the collection.
        """
        print(f"🧹 Clearing all items from collection '{self.collection_name}'...")
        # To clear a collection, we get all IDs and delete them.
        all_items = self.collection.get()
        if all_items['ids']:
            self.collection.delete(ids=all_items['ids'])
            print("✅ Collection cleared.")
        else:
            print("ℹ️ Collection was already empty.")

    def delete_collection(self):
        """Deletes the entire collection from the database."""
        print(f"💥 Deleting entire collection '{self.collection_name}'...")
        self.client.delete_collection(name=self.collection_name)
        print("✅ Collection deleted.")

    def count(self) -> int:
        """Returns the total number of items in the collection."""
        return self.collection.count()


if __name__ == "__main__":
    # --- DEMONSTRATION OF THE ChromaDBManager CLASS ---

    # 1. Initialize the manager
    db_manager = ChromaDBManager(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model_name=OLLAMA_EMBEDDING_MODEL
    )

    # 2. Add some documents (ensure Ollama is running)
    print("\n--- Step 1: Adding Documents ---")
    docs_to_add = [
        Document(page_content="The total invoice amount is INR 437.08.", metadata={"source": "invoice_summary", "page": "1"}),
        Document(page_content="An Order Cancellation Fee of INR 85.41 was charged.", metadata={"source": "invoice_details", "page": "1"}),
        Document(page_content="The bill is addressed to SHRASTI SHARMA in UTTAR PRADESH.", metadata={"source": "customer_info", "page": "1"})
    ]
    db_manager.add_documents(docs_to_add)
    print(f"Total chunks in DB: {db_manager.count()}")

    # 3. Perform a similarity search
    print("\n--- Step 2: Searching for a Document ---")
    search_results = db_manager.search(query_text="What was the cancellation fee?", n_results=1)
    if search_results and search_results['documents'][0]:
        print("✅ Closest Chunk Found:")
        print(f"   Distance (Lower is better): {search_results['distances'][0][0]:.4f}")
        print(f"   Metadata: {search_results['metadatas'][0][0]}")
        print(f"   Content: **{search_results['documents'][0][0]}**")

    # 4. Get a specific chunk by its ID (we need to generate the ID the same way)
    print("\n--- Step 3: Getting a Document by ID ---")
    doc_content_to_find = "The total invoice amount is INR 437.08."
    doc_id_to_find = hashlib.sha256(doc_content_to_find.encode('utf-8')).hexdigest()
    chunk = db_manager.get_chunk_by_id(doc_id_to_find)
    if chunk:
        print(f"✅ Found chunk by ID: {chunk}")

    # 5. Clean up the collection
    print("\n--- Step 4: Cleaning the Collection ---")
    db_manager.clear_collection()
    print(f"Total chunks in DB after clearing: {db_manager.count()}")