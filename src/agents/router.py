"""
Router Agent for AskAWS.

Classifies the incoming question into one of two routes:
  - "retrieve"  : An AWS-related question that warrants retrieval + generation
  - "refuse"    : Off-topic, harmful, or otherwise out-of-scope

Uses a small, cheap Claude call with a tightly-scoped classification prompt.
Output is constrained to a JSON object so we can parse it reliably.
"""

import os
import json
from typing import Dict
import boto3
from dotenv import load_dotenv

from src.agents.state import AgentState

load_dotenv()

GENERATION_MODEL = os.getenv("BEDROCK_GENERATION_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


# Tight system prompt: forces JSON, constrains scope
ROUTER_SYSTEM_PROMPT = """You are a routing classifier for an AWS documentation assistant.

The assistant has indexed documentation for these AWS services ONLY:
- Amazon S3 (object storage)
- AWS Lambda (serverless compute)
- Amazon EC2 (virtual machines)
- AWS IAM (identity and access management)
- Amazon SageMaker (machine learning)

Your job: classify each user question into exactly ONE of two categories:

1. "retrieve" — The question is about one of the indexed AWS services above,
   or about general AWS concepts that are likely covered (auth, networking
   basics for these services, billing/pricing concepts, etc).

2. "refuse" — The question is off-topic (not AWS), about an AWS service
   NOT in our list (e.g., DynamoDB, RDS, CloudFront), asks for harmful or
   policy-violating content, or is too vague to answer (e.g., "hi", "help").

CRITICAL: Output ONLY a JSON object with this exact structure, no other text:
{
  "route": "retrieve" | "refuse",
  "reason": "<one short sentence explaining the decision>"
}

Examples:
- "How do I create an S3 bucket?" -> {"route": "retrieve", "reason": "S3 question, in scope"}
- "What's the weather today?" -> {"route": "refuse", "reason": "Off-topic, not AWS"}
- "How do I set up DynamoDB?" -> {"route": "refuse", "reason": "DynamoDB not in indexed services"}
- "hi" -> {"route": "refuse", "reason": "Greeting, no question to answer"}
"""


class RouterAgent:
    """LLM-based classifier that decides whether to retrieve or refuse."""

    def __init__(self):
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
        )

    def _call_claude(self, question: str) -> Dict:
        """Call Claude with the classification prompt, return parsed JSON."""
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 150,          # tiny: JSON is small
            "temperature": 0.0,         # deterministic classification
            "system": ROUTER_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"Classify this question:\n\n{question}"}
            ],
        })

        response = self.bedrock.invoke_model(modelId=GENERATION_MODEL, body=body)
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"].strip()

        # Robust JSON parse — sometimes Claude wraps in markdown fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Fallback if parsing fails — default to retrieve (safer than refusing valid questions)
            return {"route": "retrieve", "reason": "Parser fallback: invalid JSON from router"}

        # Validate shape
        if parsed.get("route") not in {"retrieve", "refuse"}:
            return {"route": "retrieve", "reason": "Parser fallback: invalid route value"}

        return parsed

    def run(self, state: AgentState) -> Dict:
        """LangGraph node entrypoint. Reads state['question'], returns state delta."""
        question = state["question"]
        decision = self._call_claude(question)

        route = decision["route"]
        reason = decision["reason"]

        # Build refusal_message early so downstream graph branches don't need to
        update = {
            "route_decision": route,
            "routing_reason": reason,
            "trace": [f"[router] decision={route} reason={reason}"],
        }
        if route == "refuse":
            update["refusal_message"] = (
                f"I can only answer questions about AWS S3, Lambda, EC2, IAM, "
                f"and SageMaker. Your question was routed to refusal because: {reason}"
            )
            update["final_answer"] = update["refusal_message"]
        return update


# Module-level singleton (cheap; reused across queries)
_router_singleton = None

def router_node(state: AgentState) -> Dict:
    """LangGraph-compatible node function."""
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = RouterAgent()
    return _router_singleton.run(state)
    
