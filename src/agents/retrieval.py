"""
Retrieval Agent for AskAWS.

Embeds the user question with Titan and retrieves the top-K most similar
chunks from the FAISS index. Refactored as a LangGraph node from the
single-agent pipeline in src/rag_pipeline.py.
"""

import os
from pathlib import Path
from typing import Dict
import boto3
from dotenv import load_dotenv

from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from src.agents.state import AgentState

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"
EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))


class RetrievalAgent:
    """Embeds question and retrieves top-K chunks from FAISS."""

    def __init__(self, k: int = TOP_K):
        self.k = k
        bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
        self.embeddings = BedrockEmbeddings(client=bedrock, model_id=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            str(VECTOR_STORE_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def run(self, state: AgentState) -> Dict:
        """LangGraph node entrypoint."""
        question = state["question"]
        results = self.vectorstore.similarity_search_with_score(question, k=self.k)

        chunks = [doc for doc, _ in results]
        distances = [float(dist) for _, dist in results]

        # Get best-distance for trace
        best = min(distances) if distances else float("inf")

        return {
            "retrieved_chunks": chunks,
            "retrieved_distances": distances,
            "trace": [f"[retrieval] k={self.k} chunks={len(chunks)} best_dist={best:.3f}"],
        }


_retrieval_singleton = None

def retrieval_node(state: AgentState) -> Dict:
    """LangGraph-compatible node function."""
    global _retrieval_singleton
    if _retrieval_singleton is None:
        _retrieval_singleton = RetrievalAgent()
    return _retrieval_singleton.run(state)