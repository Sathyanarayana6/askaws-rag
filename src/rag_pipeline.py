"""
Single-agent RAG pipeline for AskAWS.

Pipeline:
  user_question
  → embed with Titan
  → retrieve top-K chunks from FAISS
  → build a grounded prompt
  → generate answer with Claude (Bedrock)
  → return answer + citations

This module exposes a single class, AskAWSRAG, with one main method: ask().
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

from dotenv import load_dotenv
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"


# ----- Configuration -----

EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
GENERATION_MODEL = os.getenv("BEDROCK_GENERATION_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))
MAX_TOKENS = 1024
TEMPERATURE = 0.2  # Low temperature: grounded, factual answers


# ----- The grounded prompt template -----

SYSTEM_PROMPT = """You are AskAWS, a focused assistant for Amazon Web Services documentation.

You answer questions strictly using the AWS documentation excerpts provided in the user message.

RULES:
1. Use ONLY the information from the provided context. Do not use prior knowledge.
2. If the context does not contain the answer, say exactly: "I don't have enough information in the AWS documentation I have indexed to answer that question."
3. Cite sources inline using [1], [2], etc. matching the numbered context items.
4. Be concise and technical. Prefer clarity over verbosity.
5. If multiple sources contribute to the answer, cite all of them.
6. Do not make up service names, API names, or feature names. Use only what appears in the context.
"""


# ----- Data classes -----

@dataclass
class Citation:
    """A single citation in a RAG response."""
    index: int
    service: str
    title: str
    source_url: str
    distance: float
    preview: str

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "service": self.service,
            "title": self.title,
            "source_url": self.source_url,
            "distance": self.distance,
            "preview": self.preview,
        }


@dataclass
class RAGResponse:
    """The full result of a single ask() call."""
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_chunks: int = 0
    refused: bool = False  # True if the model declined due to insufficient context

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_chunks": self.retrieved_chunks,
            "refused": self.refused,
        }


# ----- The core RAG class -----

class AskAWSRAG:
    """Single-agent RAG over the AskAWS FAISS knowledge base."""

    def __init__(self, k: int = TOP_K):
        self.k = k
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
        )
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock,
            model_id=EMBEDDING_MODEL,
        )
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> FAISS:
        """Load the FAISS index built in Phase 2."""
        if not VECTOR_STORE_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {VECTOR_STORE_PATH}. "
                "Run src/ingestion/build_index.py first."
            )
        return FAISS.load_local(
            str(VECTOR_STORE_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    # ----- Retrieval -----

    def retrieve(self, question: str) -> List[tuple[Document, float]]:
        """Return top-K (document, distance) pairs for a question."""
        return self.vectorstore.similarity_search_with_score(question, k=self.k)

    # ----- Prompt construction -----

    def _build_user_prompt(self, question: str, retrieved: List[tuple[Document, float]]) -> str:
        """Assemble the grounded user message from retrieved chunks."""
        context_blocks = []
        for i, (doc, _distance) in enumerate(retrieved, start=1):
            service = doc.metadata.get("service", "unknown")
            title = doc.metadata.get("title", "Untitled")
            source = doc.metadata.get("source_url", "unknown")
            block = (
                f"[{i}] Service: {service.upper()}  |  Title: {title}\n"
                f"Source: {source}\n"
                f"Content:\n{doc.page_content}\n"
            )
            context_blocks.append(block)

        context_text = "\n---\n".join(context_blocks)

        user_prompt = (
            f"# AWS Documentation Context\n\n{context_text}\n\n"
            f"# Question\n\n{question}\n\n"
            f"Answer the question using ONLY the documentation context above. "
            f"Cite sources inline as [1], [2], etc."
        )
        return user_prompt

    # ----- Generation -----

    def _call_claude(self, user_prompt: str) -> str:
        """Call Claude on Bedrock and return the text response."""
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        })

        response = self.bedrock.invoke_model(
            modelId=GENERATION_MODEL,
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    # ----- Public API -----

    def ask(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline for a single question."""
        # 1. Retrieve
        retrieved = self.retrieve(question)

        # 2. Build citations metadata (used for the response, not the prompt)
        citations = []
        for i, (doc, distance) in enumerate(retrieved, start=1):
            citations.append(Citation(
                index=i,
                service=doc.metadata.get("service", "unknown"),
                title=doc.metadata.get("title", "Untitled"),
                source_url=doc.metadata.get("source_url", "unknown"),
                distance=float(distance),
                preview=doc.page_content[:200],
            ))

        # 3. Build prompt and call Claude
        user_prompt = self._build_user_prompt(question, retrieved)
        answer = self._call_claude(user_prompt)

        # 4. Detect refusal (model says it doesn't have info)
        refused = "i don't have enough information" in answer.lower()

        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=len(retrieved),
            refused=refused,
        )


# ----- CLI entrypoint -----

def main():
    """Simple REPL: ask questions in the terminal."""
    print("=" * 70)
    print("AskAWS — Single-Agent RAG")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Top-K retrieval: {TOP_K}")
    print("Type a question. Type 'quit' or 'exit' to stop.")
    print("=" * 70)

    rag = AskAWSRAG()
    print(f"Loaded FAISS index with {rag.vectorstore.index.ntotal} vectors.\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        print("\n... thinking ...\n")
        try:
            response = rag.ask(question)
        except Exception as e:
            print(f"ERROR: {e}\n")
            continue

        print("AskAWS>")
        print(response.answer)
        print()
        print("Sources:")
        for c in response.citations:
            print(f"  [{c.index}] [{c.service.upper()}] {c.title}")
            print(f"      {c.source_url}")
            print(f"      (distance: {c.distance:.3f})")
        print()
        if response.refused:
            print("(Note: Model declined due to insufficient context.)\n")

    print("Goodbye.")


if __name__ == "__main__":
    main()
