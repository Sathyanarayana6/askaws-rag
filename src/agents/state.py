"""
Shared state for the AskAWS multi-agent graph.

Every agent reads from and writes to this state dictionary as the question
flows through the system: Router -> Retrieval -> Generator -> Critic -> Done.

LangGraph requires the state to be a TypedDict so it can validate transitions
and merge updates from each node.
"""

from typing import TypedDict, List, Optional, Literal, Annotated
from operator import add
from langchain_core.documents import Document


# Decision types the Router can produce
RouteDecision = Literal["retrieve", "refuse"]

# Verdicts the Critic can produce
CriticVerdict = Literal["pass", "fail"]


class AgentState(TypedDict, total=False):
    """Mutable state object passed between agents in the LangGraph graph.

    `total=False` means all fields are optional — agents only write what they own.

    Field ownership:
      - question:           input (set by caller)
      - route_decision:     written by Router
      - routing_reason:     written by Router (for transparency/debugging)
      - retrieved_chunks:   written by Retrieval
      - retrieved_distances: written by Retrieval (for citation metadata)
      - draft_answer:       written by Generator
      - critic_verdict:     written by Critic
      - critic_feedback:    written by Critic (why it passed/failed)
      - final_answer:       written by Critic (final answer to return)
      - refusal_message:    written by Router or Critic on failure paths
      - trace:              accumulated log of agent decisions (debug)
    """
    # Input
    question: str

    # Router outputs
    route_decision: RouteDecision
    routing_reason: str

    # Retrieval outputs
    retrieved_chunks: List[Document]
    retrieved_distances: List[float]

    # Generator output
    draft_answer: str

    # Critic outputs
    critic_verdict: CriticVerdict
    critic_feedback: str
    final_answer: str

    # Refusal handling
    refusal_message: str

    # Trace for debugging — accumulates strings across agents
    # The `Annotated[..., add]` tells LangGraph to APPEND, not overwrite
    trace: Annotated[List[str], add]


def empty_state(question: str) -> AgentState:
    """Helper to initialize a fresh state for a new question."""
    return AgentState(
        question=question,
        trace=[],
    )
