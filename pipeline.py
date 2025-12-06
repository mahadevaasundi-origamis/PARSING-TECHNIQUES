import os
from Rags.chunker import DocumentChunker
from Rags.embeddding_generator import EmbeddingGenerator

# --- Master Configuration for the Pipeline ---

# Input for the entire pipeline (raw, unchunked data)
PIPELINE_INPUT_PATH = "d:/GitHub/Learning/PARSING-TECHNIQUES/merged_output.json"

# Intermediate file produced by the chunker and consumed by the embedder
PROCESSED_CHUNKS_PATH = "d:/GitHub/Learning/PARSING-TECHNIQUES/output_processed.json"

# Final destination for the vector database
CHROMA_PERSIST_DIRECTORY = "d:/GitHub/Learning/PARSING-TECHNIQUES/Rags/chroma_db"
COLLECTION_NAME = "document_chunks_collection"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


def run_pipeline():
    """
    Executes the complete data ingestion pipeline:
    1. Chunks the source documents.
    2. Validates the chunks.
    3. Generates embeddings and stores them in ChromaDB.
    """
    print("🚀 --- Starting Data Ingestion Pipeline --- 🚀")

    # --- Step 1: Chunking Documents ---
    print("\n[STEP 1/2] Chunking source documents...")
    if not os.path.exists(PIPELINE_INPUT_PATH):
        print(f"❌ FATAL: Pipeline input file not found at {PIPELINE_INPUT_PATH}")
        return

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunker.process_file(PIPELINE_INPUT_PATH, PROCESSED_CHUNKS_PATH)
    print("✅ Chunking complete.")

    # --- Step 2: Generating Embeddings and Storing in DB ---
    print("\n[STEP 2/2] Generating embeddings and storing in ChromaDB...")
    # This generator now uses our ChromaDBManager internally
    embedding_pipeline = EmbeddingGenerator(
        input_path=PROCESSED_CHUNKS_PATH,
        db_path=CHROMA_PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        model_name=OLLAMA_EMBEDDING_MODEL
    )
    embedding_pipeline.run()

    print("\n🎉 --- Pipeline Finished Successfully! --- 🎉")

if __name__ == "__main__":
    run_pipeline()