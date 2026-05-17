"""
Test retrieval against the FAISS index.

Runs a handful of representative questions and prints the top-K retrieved chunks
with their similarity scores. This is the moment we prove the knowledge base works.

Usage:
    python scripts/test_retrieval.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
import boto3

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"
EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))

# Test queries spanning all 5 services
TEST_QUERIES = [
    "How do I create an S3 bucket?",
    "What is the difference between EC2 on-demand and spot instances?",
    "How do IAM roles work with EC2 instances?",
    "How do I trigger a Lambda function from S3?",
    "What is SageMaker Feature Store?",
    "How do I encrypt data at rest in S3?",
    "What are Lambda cold starts?",
    "How do I attach an IAM policy to a role?",
]


def load_vectorstore():
    """Load FAISS index from disk with the same embedder used to build it."""
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=EMBEDDING_MODEL,
    )
    vectorstore = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we created the index ourselves
    )
    return vectorstore


def run_query(vectorstore, query: str, k: int = TOP_K):
    """Retrieve top-K chunks for a query and print them with scores."""
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    # similarity_search_with_score returns (Document, distance) pairs
    # Lower distance = more similar (L2 distance)
    results = vectorstore.similarity_search_with_score(query, k=k)

    for i, (doc, distance) in enumerate(results, 1):
        service = doc.metadata.get("service", "?")
        title = doc.metadata.get("title", "?")
        source = doc.metadata.get("source_url", "?")
        chunk_idx = doc.metadata.get("chunk_index", "?")

        print(f"\n[{i}] [{service.upper()}] {title}")
        print(f"    Distance: {distance:.4f}  |  Chunk: {chunk_idx}")
        print(f"    Source:   {source}")
        print(f"    Preview:  {doc.page_content[:250]}...")
    print()


def main():
    print("Loading FAISS index from disk...")
    vectorstore = load_vectorstore()
    print(f"  Loaded {vectorstore.index.ntotal} vectors\n")

    for query in TEST_QUERIES:
        run_query(vectorstore, query)


if __name__ == "__main__":
    main()