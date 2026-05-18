"""
LangGraph orchestration for AskAWS multi-agent RAG.

Builds a StateGraph that wires together the 4 agents:
  Router -> (Retrieval -> Generator -> Critic) -> END
       \-> END (on refusal)

The router decides whether to do full retrieval+generation+critic, or to
short-circuit on out-of-scope queries.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.router import router_node
from src.agents.retrieval import retrieval_node
from src.agents.generator import generator_node
from src.agents.critic import critic_node


def route_from_router(state: AgentState) -> Literal["retrieve", "refuse"]:
    """Conditional edge: read the router's decision and pick the next node.

    LangGraph calls this function after the router node runs to decide
    which edge to follow.
    """
    return state.get("route_decision", "retrieve")


def build_graph():
    """Construct and compile the AskAWS multi-agent graph."""
    graph = StateGraph(AgentState)

    # Add the 4 agent nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edge from router based on its decision
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {
            "retrieve": "retrieval",  # in-scope question -> do full RAG
            "refuse": END,            # out-of-scope -> short-circuit
        },
    )

    # Linear chain for the retrieval path
    graph.add_edge("retrieval", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


# Module-level compiled graph (built once, reused across queries)
_graph_singleton = None


def get_graph():
    """Get the compiled graph (lazy singleton)."""
    global _graph_singleton
    if _graph_singleton is None:
        _graph_singleton = build_graph()
    return _graph_singleton


def ask(question: str) -> dict:
    """High-level entry point. Runs a question through the full graph.

    Returns the final state dictionary, which includes:
      - final_answer
      - retrieved_chunks (if retrieval ran)
      - retrieved_distances (if retrieval ran)
      - route_decision, critic_verdict (if critic ran)
      - trace (debug log)
    """
    graph = get_graph()
    initial_state = {"question": question, "trace": []}
    final_state = graph.invoke(initial_state)
    return final_state

def ask_streaming(question: str):
    """Generator that yields each agent step as it completes.

    Yields tuples of (agent_name, partial_state_update).
    Final state is yielded under the key "final".

    Use this for UIs that want to show live progress.
    """
    graph = get_graph()
    initial_state = {"question": question, "trace": []}

    accumulated = dict(initial_state)
    for chunk in graph.stream(initial_state):
        # chunk is a dict like {"router": {...partial state update...}}
        for node_name, update in chunk.items():
            # Merge the update into accumulated state (trace appends, others overwrite)
            for key, value in update.items():
                if key == "trace":
                    accumulated.setdefault("trace", []).extend(value)
                else:
                    accumulated[key] = value
            yield node_name, dict(accumulated)

    # Final yield with everything assembled
    yield "final", accumulated