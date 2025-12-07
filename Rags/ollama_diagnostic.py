#!/usr/bin/env python3
"""
Diagnostic script to verify Ollama embeddings are working correctly.
Run this before processing documents to ensure your setup is complete.
"""

import sys
import subprocess
import time
from langchain_community.embeddings import OllamaEmbeddings
import requests

def check_ollama_running():
    """Check if Ollama service is running"""
    print("🔍 Checking if Ollama is running...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running on http://localhost:11434")
            return True
        else:
            print(f"⚠️  Ollama returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama on http://localhost:11434")
        print("   Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def list_available_models():
    """List available Ollama models"""
    print("\n🔍 Checking available models...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get("models", [])
        
        if not models:
            print("⚠️  No models available")
            return []
        
        print(f"✅ Found {len(models)} model(s):")
        for model in models:
            name = model.get("name", "unknown")
            print(f"   • {name}")
        
        return [m.get("name") for m in models]
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []


def check_embedding_model(model_name="nomic-embed-text"):
    """Check if embedding model is available"""
    print(f"\n🔍 Checking for embedding model: {model_name}")
    try:
        models = requests.get("http://localhost:11434/api/tags").json().get("models", [])
        model_names = [m.get("name") for m in models]
        
        # Check both with and without :latest tag
        model_variants = [model_name, f"{model_name}:latest"]
        
        for variant in model_variants:
            if variant in model_names:
                print(f"✅ Model '{variant}' is available")
                return True, variant  # Return the actual model name found
        
        print(f"❌ Model '{model_name}' not found")
        print(f"\n   To install it manually, run in terminal:")
        print(f"   ollama pull {model_name}")
        print(f"\n   Available models: {', '.join(model_names)}")
        return False, None
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        return False, None


def test_embedding_generation(model_name="nomic-embed-text"):
    """Test if embeddings can be generated"""
    print(f"\n🔍 Testing embedding generation with {model_name}...")
    try:
        embedding_fn = OllamaEmbeddings(model=model_name)
        
        test_text = "Cortex-M3 processor architecture"
        print(f"   Testing with: '{test_text}'")
        
        embedding = embedding_fn.embed_query(test_text)
        
        print(f"✅ Embedding generated successfully!")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        return True, embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False, None


def test_batch_embeddings(model_name="nomic-embed-text"):
    """Test batch embedding generation"""
    print(f"\n🔍 Testing batch embedding generation...")
    try:
        embedding_fn = OllamaEmbeddings(model=model_name)
        
        test_texts = [
            "Cortex-M3 processor",
            "ARM architecture",
            "Embedded systems"
        ]
        
        print(f"   Testing with {len(test_texts)} texts...")
        embeddings = embedding_fn.embed_documents(test_texts)
        
        print(f"✅ Batch embedding generated successfully!")
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   Each embedding dimension: {len(embeddings[0])}")
        
        return True
    except Exception as e:
        print(f"❌ Error in batch embedding: {e}")
        return False


def diagnose_chromadb(persist_dir="d:/GitHub/Learning/PARSING-TECHNIQUES/Rags/chroma_db"):
    """Diagnose ChromaDB setup"""
    print(f"\n🔍 Checking ChromaDB configuration...")
    try:
        import chromadb
        from chromadb.config import Settings
        
        print(f"   Persist directory: {persist_dir}")
        
        # Try to connect
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        
        print(f"✅ ChromaDB connection successful!")
        print(f"   Collections found: {len(collections)}")
        
        for collection in collections:
            count = collection.count()
            print(f"   • {collection.name}: {count} documents")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False


def main():
    print("=" * 60)
    print("🔧 Ollama Embeddings Diagnostic")
    print("=" * 60)
    
    results = {
        "ollama_running": False,
        "embedding_model_available": False,
        "embeddings_working": False,
        "batch_embeddings_working": False,
        "chromadb_working": False
    }
    
    actual_model_name = "nomic-embed-text:latest"  # Default
    
    # Check 1: Ollama running
    if not check_ollama_running():
        print("\n❌ CRITICAL: Ollama is not running!")
        print("   Start it with: ollama serve")
        return results
    results["ollama_running"] = True
    
    # Check 2: List models
    models = list_available_models()
    
    # Check 3: Check embedding model
    model_found, found_model_name = check_embedding_model("nomic-embed-text")
    if found_model_name:
        actual_model_name = found_model_name
    results["embedding_model_available"] = model_found
    
    if not model_found:
        print("\n⚠️  IMPORTANT: Run this command manually:")
        print("   ollama pull nomic-embed-text")
        print("\n   Then run this diagnostic again.")
    
    # Check 4: Test single embedding
    if model_found:
        success, embedding = test_embedding_generation(actual_model_name)
        results["embeddings_working"] = success
        
        # Check 5: Test batch embeddings
        if success:
            results["batch_embeddings_working"] = test_batch_embeddings(actual_model_name)
    
    # Check 6: Test ChromaDB
    results["chromadb_working"] = diagnose_chromadb()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Diagnostic Summary")
    print("=" * 60)
    
    status_icon = {True: "✅", False: "❌"}
    for check, result in results.items():
        icon = status_icon[result]
        print(f"{icon} {check.replace('_', ' ').title()}: {'OK' if result else 'FAILED'}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed! Your system is ready.")
        print("\nYou can now:")
        print("1. Upload a document: POST /upload/")
        print("2. Process it: POST /process/{file_id}")
        print("3. Query it: GET /qa/?query=your+question")
    else:
        if not results["embedding_model_available"]:
            print("❌ Main issue: Embedding model not available")
            print("\n📝 FIX:")
            print("1. Open Terminal/CMD")
            print("2. Run: ollama pull nomic-embed-text")
            print("3. Wait for download to complete")
            print("4. Run this diagnostic again")
        else:
            print("❌ Some checks failed. Please fix the issues above.")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()