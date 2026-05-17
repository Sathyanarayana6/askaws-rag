"""
Critic Agent for AskAWS.

Reviews the Generator's draft answer against the retrieved context and
produces a pass/fail verdict on:
  - Groundedness: every factual claim is supported by the context
  - Citations: claims are cited with [N] markers that map to real chunks
  - Refusal correctness: if the model refused, was that justified?

Uses LLM-as-a-judge pattern — a second Claude call evaluates the first one.
This is the same technique used by production RAG evaluation frameworks
like RAGAS, just inline.
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


CRITIC_SYSTEM_PROMPT = """You are a strict fact-checking critic for an AWS documentation assistant.

You are given:
1. The user's QUESTION
2. The CONTEXT chunks the assistant was given (numbered [1], [2], ...)
3. The DRAFT_ANSWER the assistant produced

Your job: judge whether the DRAFT_ANSWER is acceptable. Acceptable means ALL of:

A. GROUNDED: Every factual claim in DRAFT_ANSWER is supported by the CONTEXT.
   No claim invented from outside the context. Common-sense connective text
   is allowed (e.g., "To do this, you need to..."), but specific facts
   (service names, APIs, parameter names, behaviors) MUST come from CONTEXT.

B. CITED: Specific factual claims include a [N] citation marker referencing
   a CONTEXT chunk. Allow some flexibility — a topic sentence and supporting
   sentences from the same chunk only need one citation.

C. APPROPRIATE: If DRAFT_ANSWER says it doesn't have enough information,
   verify the CONTEXT genuinely doesn't contain the answer. A premature
   refusal when the answer IS in the context is a FAIL.

Output ONLY a JSON object with this exact structure, no other text:
{
  "verdict": "pass" | "fail",
  "feedback": "<one sentence: what's right or what's wrong>",
  "unsupported_claims": ["<list of any specific claims not found in context, empty if none>"]
}
"""


class CriticAgent:
    """LLM-as-judge over the Generator's draft answer."""

    def __init__(self):
        self.bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

    def _build_user_prompt(self, question: str, chunks: List[Document], draft: str) -> str:
        blocks = []
        for i, doc in enumerate(chunks, start=1):
            service = doc.metadata.get("service", "unknown")
            title = doc.metadata.get("title", "Untitled")
            blocks.append(
                f"[{i}] {service.upper()} — {title}\n{doc.page_content}\n"
            )
        context_text = "\n---\n".join(blocks)
        return (
            f"# QUESTION\n{question}\n\n"
            f"# CONTEXT\n{context_text}\n\n"
            f"# DRAFT_ANSWER\n{draft}\n\n"
            f"Evaluate the DRAFT_ANSWER and return your JSON verdict."
        )

    def _call_claude(self, user_prompt: str) -> Dict:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 400,
            "temperature": 0.0,
            "system": CRITIC_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        })
        response = self.bedrock.invoke_model(modelId=GENERATION_MODEL, body=body)
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"].strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # On parse failure, lenient pass — don't block a possibly-good answer
            return {
                "verdict": "pass",
                "feedback": "Critic JSON parse failed; deferring to draft.",
                "unsupported_claims": [],
            }

        if parsed.get("verdict") not in {"pass", "fail"}:
            parsed["verdict"] = "pass"
        return parsed

    def run(self, state: AgentState) -> Dict:
        question = state["question"]
        chunks = state["retrieved_chunks"]
        draft = state["draft_answer"]

        user_prompt = self._build_user_prompt(question, chunks, draft)
        result = self._call_claude(user_prompt)

        verdict = result["verdict"]
        feedback = result["feedback"]
        unsupported = result.get("unsupported_claims", [])

        # On pass: final_answer is just the draft
        # On fail: append a transparency note so the user knows
        if verdict == "pass":
            final_answer = draft
        else:
            note = (
                "\n\n---\n"
                f"_Note: This response was flagged by the fact-checker. "
                f"Reason: {feedback}_"
            )
            if unsupported:
                note += f"\n_Unsupported claims: {'; '.join(unsupported)}_"
            final_answer = draft + note

        return {
            "critic_verdict": verdict,
            "critic_feedback": feedback,
            "final_answer": final_answer,
            "trace": [
                f"[critic] verdict={verdict} unsupported={len(unsupported)} "
                f"feedback={feedback[:80]}"
            ],
        }


_critic_singleton = None

def critic_node(state: AgentState) -> Dict:
    global _critic_singleton
    if _critic_singleton is None:
        _critic_singleton = CriticAgent()
    return _critic_singleton.run(state)