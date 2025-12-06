# --- Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
CHROMA_PERSIST_DIRECTORY = "d:/GitHub/Learning/PARSING-TECHNIQUES/Rags/chroma_db"
COLLECTION_NAME = "document_chunks_collection"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "gpt-oss:120b-cloud"


class QASystem:
    """
    Encapsulates the logic for a Retrieval-Augmented Generation (RAG) system.
    It loads a vector store, sets up a retriever, and creates a question-answering chain.
    """

    def __init__(self, db_path, collection_name, embedding_model, llm_model):
        print("--- Initializing RAG QA System ---")

        # 1. Initialize Embeddings
        embedding_function = OllamaEmbeddings(model=embedding_model)

        # 2. Load the Chroma Vector Store
        self.vector_store = Chroma(
            persist_directory=db_path,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
        print(f"✅ Vector store loaded from '{db_path}' with {self.vector_store._collection.count()} documents.")

        # 3. Create a Retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        print("✅ Retriever created.")

        # 4. Define the RAG Prompt Template
        rag_template = """Answer the user's question based solely on the following context.
If the context does not contain the answer, state that you cannot find the information in the provided documents.

<context>
{context}
</context>

Question: {input}
"""
        rag_prompt = ChatPromptTemplate.from_template(rag_template)

        # 5. Initialize the LLM
        llm = ChatOllama(model=llm_model)
        print(f"✅ LLM model '{llm_model}' initialized.")

        # 6. Create the RAG Chain using LCEL
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "input": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        print("✅ RAG chain created and ready.")

    def ask_question(self, question: str):
        """Asks a question to the RAG chain and prints the results."""
        print(f"\n❓ Querying the system with: '{question}'")
        
        # Get context separately for display
        retrieved_docs = self.retriever.invoke(question)
        
        print("\n--- Retrieved Context Chunks ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"Chunk {i+1}:\n{doc.page_content}\n")
            print(f"Source: {doc.metadata.get('sourcepage', 'N/A')}\n---")

        # Get answer
        answer = self.rag_chain.invoke(question)
        
        print("\n🤖 Generated Answer:")
        print(answer)


if __name__ == "__main__":
    qa_system = QASystem(CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, OLLAMA_EMBEDDING_MODEL, OLLAMA_LLM_MODEL)
    qa_system.ask_question("What is the total invoice amount?")