"""
Document chunker: splits cleaned AWS Markdown docs into retrieval chunks.

Uses LangChain's RecursiveCharacterTextSplitter which respects natural
boundaries (paragraphs, sentences, words) when splitting.

Each chunk preserves source metadata for citation in RAG responses.
"""

import os
import re
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Settings from .env (with safe defaults)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_frontmatter(content: str) -> tuple[Dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body_text)."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    metadata = {}
    for line in frontmatter_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata, body


def load_documents(processed_dir: Path = PROCESSED_DIR) -> List[Document]:
    """Walk processed/ folder and load each .md as a LangChain Document."""
    documents = []
    md_files = list(processed_dir.rglob("*.md"))

    print(f"Loading {len(md_files)} markdown files...")

    for filepath in md_files:
        try:
            content = filepath.read_text(encoding="utf-8")
            metadata, body = parse_frontmatter(content)

            # Add filepath info for traceability
            metadata["filepath"] = str(filepath.relative_to(PROJECT_ROOT))
            # Service comes from parent folder name (s3, lambda, ec2, iam, sagemaker)
            if "service" not in metadata:
                metadata["service"] = filepath.parent.name

            documents.append(Document(page_content=body, metadata=metadata))
        except Exception as e:
            print(f"  Failed to load {filepath}: {e}")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into retrieval chunks using recursive character splitter.

    The splitter tries separators in order: paragraphs, lines, sentences,
    words, characters. This respects natural document structure.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    MIN_CHUNK_CHARS = 100  # Drop chunks shorter than this (titles, fragments)

    chunks = []
    dropped = 0
    for doc in documents:
        # Split into chunks (LangChain preserves metadata on each chunk)
        doc_chunks = splitter.split_documents([doc])

        # Filter out tiny chunks
        useful_chunks = [c for c in doc_chunks if len(c.page_content) >= MIN_CHUNK_CHARS]
        dropped += len(doc_chunks) - len(useful_chunks)

        # Add chunk-level metadata (re-index after filtering)
        for i, chunk in enumerate(useful_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(useful_chunks)
        chunks.extend(useful_chunks)

    print(f"  Dropped {dropped} chunks below {MIN_CHUNK_CHARS} chars (titles/fragments)")
    return chunks


def main():
    print("=" * 60)
    print("AskAWS Document Chunker")
    print(f"Chunk size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP}")
    print("=" * 60)

    documents = load_documents()
    print(f"  Loaded {len(documents)} documents")

    # Print stats per service
    from collections import Counter
    service_counts = Counter(d.metadata.get("service", "unknown") for d in documents)
    for service, count in sorted(service_counts.items()):
        print(f"    {service}: {count} docs")

    print()
    print("Chunking...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks total")
    print(f"  Average chunks per doc: {len(chunks) / len(documents):.1f}")

    # Sample a few chunks
    print()
    print("Sample chunks:")
    print("-" * 60)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nCHUNK {i + 1} (from {chunk.metadata.get('title', '?')}):")
        print(f"  Service: {chunk.metadata.get('service')}")
        print(f"  Length:  {len(chunk.page_content)} chars")
        print(f"  Preview: {chunk.page_content[:200]}...")

    # Stats
    chunk_lengths = [len(c.page_content) for c in chunks]
    print()
    print("Chunk length stats:")
    print(f"  Min:  {min(chunk_lengths)}")
    print(f"  Max:  {max(chunk_lengths)}")
    print(f"  Avg:  {sum(chunk_lengths) // len(chunk_lengths)}")
    print()

    return chunks


if __name__ == "__main__":
    main()