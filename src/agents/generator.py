"""
Generator Agent for AskAWS.

Takes retrieved chunks + question and produces a grounded, cited answer
using Claude Haiku 4.5 on Bedrock.
"""

import os
import json
from typing import Dict, List
import boto3
from dotenv import load_dotenv
from langchain.schema import Document

from src.agents.state import AgentState

load_dotenv()

GENERATION_MODEL = os.getenv("BEDROCK_GENERATION_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MAX_TOKENS = 1024
TEMPERATURE = 0.2


SYSTEM_PROMPT = """You are AskAWS, a focused assistant for Amazon Web Services documentation.

You answer questions strictly using the AWS documentation excerpts provided.

RULES:
1. Use ONLY information from the provided context. Do not use prior knowledge.
2. If the context does not contain the answer, say exactly:
   "I don't have enough information in the AWS documentation I have indexed to answer that question."
3. Cite sources inline using [1], [2], etc. matching the numbered context items.
4. Be concise and technical. Prefer clarity over verbosity.
5. If multiple sources contribute, cite all of them.
6. Do not make up service names, APIs, or features. Use only what appears in the context.
"""


class GeneratorAgent:
    """Generates a grounded answer from retrieved context."""

    def __init__(self):
        self.bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

    def _build_user_prompt(self, question: str, chunks: List[Document]) -> str:
        blocks = []
        for i, doc in enumerate(chunks, start=1):
            service = doc.metadata.get("service", "unknown")
            title = doc.metadata.get("title", "Untitled")
            source = doc.metadata.get("source_url", "unknown")
            blocks.append(
                f"[{i}] Service: {service.upper()}  |  Title: {title}\n"
                f"Source: {source}\n"
                f"Content:\n{doc.page_content}\n"
            )
        context_text = "\n---\n".join(blocks)
        return (
            f"# AWS Documentation Context\n\n{context_text}\n\n"
            f"# Question\n\n{question}\n\n"
            f"Answer using ONLY the context above. Cite sources inline as [1], [2], etc."
        )

    def run(self, state: AgentState) -> Dict:
        """LangGraph node entrypoint."""
        question = state["question"]
        chunks = state["retrieved_chunks"]

        user_prompt = self._build_user_prompt(question, chunks)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        })
        response = self.bedrock.invoke_model(modelId=GENERATION_MODEL, body=body)
        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]

        return {
            "draft_answer": answer,
            "trace": [f"[generator] answer_len={len(answer)} chars"],
        }


_generator_singleton = None

def generator_node(state: AgentState) -> Dict:
    global _generator_singleton
    if _generator_singleton is None:
        _generator_singleton = GeneratorAgent()
    return _generator_singleton.run(state)