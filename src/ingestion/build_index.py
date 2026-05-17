"""
Build the FAISS vector index from chunked AWS documentation.

Pipeline:
  Markdown files → chunks → Bedrock Titan embeddings → FAISS index on disk

This is the most expensive step in terms of API calls (still very cheap with Titan).
Run it ONCE. The resulting index is saved to data/vector_store/.

Usage:
    python src/ingestion/build_index.py
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import boto3

# Make chunker importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunker import load_documents, chunk_documents

load_dotenv()

# Config from .env
EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"
BATCH_SIZE = 50  # Embed chunks in batches to manage rate limits


def embed_chunks_in_batches(chunks: List[Document], embeddings) -> FAISS:
    """Embed chunks in batches and build FAISS index incrementally.

    Why batches?
    - Bedrock has per-second rate limits
    - Failure on chunk 1500 of 2000 would waste prior work without batching
    - Allows progress reporting
    """
    print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Region: {AWS_REGION}")
    print()

    vectorstore = None
    failed_count = 0
    start_time = time.time()

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Batches"):
        batch = chunks[i:i + BATCH_SIZE]
        try:
            if vectorstore is None:
                # First batch creates the index
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                # Subsequent batches add to existing index
                batch_store = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_store)
        except Exception as e:
            print(f"\n  Batch {i // BATCH_SIZE + 1} failed: {e}")
            failed_count += len(batch)
            # Small back-off then continue
            time.sleep(2)
            continue

    elapsed = time.time() - start_time
    print(f"\nEmbedding complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if failed_count:
        print(f"  WARN: {failed_count} chunks failed (continued past errors)")

    return vectorstore


def save_index(vectorstore: FAISS, save_path: Path):
    """Persist FAISS index + a metadata sidecar for reproducibility."""
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))

    # Save metadata so we know what was indexed
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "region": AWS_REGION,
        "num_vectors": vectorstore.index.ntotal,
        "embedding_dim": vectorstore.index.d,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_path / "index_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nIndex saved to: {save_path}")
    print(f"  Vectors:       {metadata['num_vectors']}")
    print(f"  Dimension:     {metadata['embedding_dim']}")
    print(f"  Files written: index.faiss, index.pkl, index_metadata.json")


def main():
    print("=" * 60)
    print("AskAWS: Building FAISS Vector Index")
    print("=" * 60)

    # Step 1: Load and chunk documents
    print("\n[1/3] Loading and chunking documents...")
    documents = load_documents()
    chunks = chunk_documents(documents)
    print(f"  Ready to embed {len(chunks)} chunks")

    # Step 2: Initialize Bedrock embeddings client
    print("\n[2/3] Initializing Bedrock embeddings client...")
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=EMBEDDING_MODEL,
    )
    # Test one embedding call to fail fast on auth issues
    try:
        test_vec = embeddings.embed_query("test")
        print(f"  Test embedding OK (dim={len(test_vec)})")
    except Exception as e:
        print(f"  FATAL: cannot embed. Check AWS credentials and model access.")
        print(f"  {e}")
        return

    # Step 3: Embed all chunks and build FAISS
    print("\n[3/3] Embedding chunks and building FAISS index...")
    vectorstore = embed_chunks_in_batches(chunks, embeddings)

    if vectorstore is None:
        print("ERROR: No vectors generated. Aborting.")
        return

    # Save index to disk
    save_index(vectorstore, VECTOR_STORE_PATH)

    print("\n" + "=" * 60)
    print("DONE! Vector index ready for retrieval.")
    print("=" * 60)


if __name__ == "__main__":
    main()