# AskAWS — Multi-Agent RAG Assistant for AWS Documentation

> A friendly assistant that helps you learn AWS. Four AI agents work together to answer your questions with grounded, cited responses from official AWS documentation.

**Live demo:** [askaws.streamlit.app](https://askaws.streamlit.app)

**Built by:** [Sathya Balla](https://www.linkedin.com/in/sathyanarayana-balla-3888721b7)
---

## What This Is

AskAWS is a production-pattern Retrieval-Augmented Generation (RAG) system built to help students and junior engineers learn AWS. It indexes **1,893 chunks** from official AWS documentation across 5 services — S3, Lambda, EC2, IAM, and SageMaker — and answers natural-language questions with cited, fact-checked responses.

Unlike a generic ChatGPT response about AWS, every answer here is:
- **Grounded** in actual AWS documentation
- **Cited** with links to the source pages
- **Fact-checked** by a Critic agent before being returned
- **Refused** when the question is outside the knowledge base (no hallucination)

---

## Why I Built It

Most ChatGPT-style AWS answers mix real facts with confidently wrong details. A new developer can't tell the difference. AskAWS solves this with a **multi-agent architecture** that grounds answers in source documents and validates them before delivery — the same pattern enterprises use for their internal knowledge assistants.

This project demonstrates production RAG engineering: streaming ingestion, vector search, multi-agent orchestration, LLM-as-judge evaluation, and automated quality metrics with RAGAS.

---

## Architecture

```
                   ┌─────────────────┐
       Question ──▶│  Router Agent   │── refuse ──▶ End
                   └────────┬────────┘
                            │ retrieve
                            ▼
                   ┌─────────────────┐
                   │ Retrieval Agent │  FAISS top-5 search
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Generator Agent │  Claude grounded answer
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Critic Agent   │  LLM-as-judge fact-check
                   └────────┬────────┘
                            │
                            ▼
                       Final Answer
```
**Four specialized agents** orchestrated as a LangGraph StateGraph with conditional routing:

| Agent | Job | Why It Matters |
|---|---|---|
| **Router** | Classifies if the question is in-scope | Saves cost — refuses off-topic queries before retrieval/generation |
| **Retrieval** | Embeds query, searches FAISS top-5 | Brings the right context to the LLM |
| **Generator** | Writes grounded answer with citations | Constrained to use only retrieved chunks |
| **Critic** | Reviews answer for groundedness | Catches hallucinations via LLM-as-judge |

---

## Stack

| Layer | Tool |
|---|---|
| **LLM (generation, routing, critic)** | Anthropic Claude Haiku 4.5 via Amazon Bedrock |
| **Embeddings** | Amazon Titan Embeddings v2 (1024-dim) |
| **Vector store** | FAISS (local, in-memory) |
| **Agent orchestration** | LangGraph |
| **LLM framework** | LangChain |
| **UI** | Streamlit |
| **Evaluation** | RAGAS (faithfulness, answer relevancy, context precision, context recall) |
| **Language** | Python 3.12 |

---

## Evaluation Results

Evaluated on a **29-question golden test set** spanning all 5 services. RAGAS metrics were computed using Bedrock Claude as the judge model (not OpenAI — full AWS-native stack).

### Overall

| Metric | Score | What It Measures |
|---|---|---|
| **Faithfulness** | **0.85** | Every factual claim in the answer is supported by retrieved context |
| **Answer Relevancy** | **0.80** | The answer actually addresses the question asked |
| **Context Precision** | **0.74** | The right chunks were retrieved for the question |
| **Context Recall** | **0.64** | Retrieval found enough info to fully answer the question |

### Per-Service Breakdown

| Service | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| **S3** | 1.00 | 0.91 | 0.62 | 0.76 |
| **Lambda** | 0.92 | 0.91 | 0.97 | 0.76 |
| **EC2** | n/a* | 0.96 | 0.88 | 0.53 |
| **IAM** | 0.75 | 0.56 | 0.60 | 0.61 |
| **SageMaker** | 0.38 | 0.73 | 0.64 | 0.53 |

*EC2 faithfulness encountered RAGAS parser errors on long technical specifications; full data in `data/evaluation_runs/`.

### Honest Engineering Notes

Strong results on S3 (perfect faithfulness) and Lambda (best precision). SageMaker scored notably lower, and the reasons are documented engineering trade-offs rather than system failures:

1. **Chunk fragmentation on procedural content.** SageMaker docs describe multi-step workflows (training jobs, pipelines, model registry). Cutting these at 800 chars sometimes splits procedures, reducing faithfulness even when the generator produces correct AWS knowledge.
2. **Generic ground-truth answers.** Some test set answers are concise summaries while the model generates more detailed, specific responses — RAGAS faithfulness penalizes anything not literally in the context.

Both are typical RAG evaluation trade-offs that production teams document and tune for. v2 plans address them via semantic chunking and richer ground-truth annotations.

Raw evaluation data is reproducible — see `data/evaluation_runs/` after running the eval scripts.

---

## How It Works

### 1. Data Ingestion

- Fetched 150 curated AWS documentation pages from `docs.aws.amazon.com` using LangChain's `WebBaseLoader`
- Cleaned HTML → Markdown with YAML frontmatter for traceability
- Chunked with `RecursiveCharacterTextSplitter` (800 chars, 100 overlap)
- Filtered chunks under 100 chars (titles, fragments)
- Embedded with Titan v2 → **1,893 vectors at 1024 dimensions**
- Persisted to FAISS index on disk

### 2. Multi-Agent Pipeline

Each question flows through a LangGraph `StateGraph` with:

- A typed `AgentState` for shared state
- Conditional edges from the Router (`retrieve` vs `refuse`)
- Linear chain through Retrieval → Generator → Critic
- Streaming support for live UI updates

### 3. Evaluation

- Golden set: 30 hand-curated Q&A pairs (6 per service)
- Automated scoring with RAGAS on 4 dimensions
- Reproducible end-to-end: `python src/evaluation/run_evaluation.py` then `python src/evaluation/score_with_ragas.py`

---
---

## Setup

### Prerequisites

- Python 3.10+
- AWS account with Bedrock model access for Claude Haiku 4.5 and Titan Embeddings v2
- IAM user with `AmazonBedrockFullAccess` permission

### Install

```bash
git clone https://github.com/Sathyanarayana6/askaws-rag.git
cd askaws-rag

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1       # Windows PowerShell
# source venv/bin/activate        # macOS/Linux

# For running the app:
pip install -r requirements.txt

# For development (running tests + evaluation):
pip install -r requirements-dev.txt
```

### Configure

```bash
cp .env.example .env
# Open .env and replace placeholders with your actual AWS credentials
```

### Build the Knowledge Base

```bash
# Step 1: Fetch AWS documentation (~3 min, ~150 URLs)
python scripts/fetch_docs.py

# Step 2: Chunk, embed, and build the FAISS index (~5 min)
python src/ingestion/build_index.py
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Run the Evaluation

```bash
# Step 1: Run all 30 questions through the multi-agent graph (~3 min)
python src/evaluation/run_evaluation.py

# Step 2: Score the answers with RAGAS (~10 min)
python src/evaluation/score_with_ragas.py
```

---

## Engineering Decisions

A few design choices worth highlighting (and good interview talking points):

- **Router-first architecture.** Off-topic queries are refused at the Router stage, saving the cost of retrieval + generation for every junk query. In production this matters for cost control at scale.
- **LLM-as-judge critic.** Rather than ship every answer blindly, a second LLM pass validates groundedness. Same pattern used by production RAG evaluation tools like RAGAS, applied inline.
- **Stateless single-turn.** Each question is answered independently — no conversation memory. This keeps citations clean and prevents context pollution across turns. Multi-turn memory is a roadmap item.
- **Bedrock-native evaluation.** RAGAS defaults to OpenAI. I rewired it to use Bedrock Claude + Titan so the entire stack stays on AWS — what an AWS-focused production team would do.
- **Curated URL list over scraping.** Used 150 hand-picked URLs via `WebBaseLoader` instead of crawling. Better data quality, no ToS gray area, faster builds.
- **Custom Streamlit UI inside columns.** Replaced default `st.chat_input` (full page-width) with a `st.form` + `st.text_input` inside the chat column for clean split-screen layout with the live agent visualization.

---

## Tech Highlights

- ✅ 4-agent system with conditional routing (LangGraph `StateGraph`)
- ✅ Production-style RAG with grounding, citations, and fact-checking
- ✅ Automated evaluation with industry-standard RAGAS metrics
- ✅ Bedrock-native stack (Claude Haiku 4.5 + Titan v2 + Bedrock for evaluation)
- ✅ Streaming UI showing live agent state transitions
- ✅ Reproducible builds (URLs in YAML, golden set in YAML, deterministic chunking)
- ✅ Clean Git history with phase-based commits documenting the build

---

## What's Next (v2 Roadmap)

- **Better retrieval.** Hybrid search (BM25 + dense), query rewriting, re-ranking with Cohere Rerank
- **Conversation memory.** Multi-turn context with summary buffers
- **Larger knowledge base.** Add DynamoDB, RDS, CloudFront, VPC, ECS, CloudWatch, API Gateway
- **Streaming generation.** Token-by-token output in the UI
- **Better SageMaker chunks.** Semantic chunking respecting procedural boundaries
- **Guardrails.** Amazon Bedrock Guardrails for prompt injection defense
- **Cost monitoring dashboard.** Per-query token and dollar cost tracking

---

## Cost

Running this project costs approximately:

- **Index build (one-time):** ~$0.02 (Titan embeddings for 1,893 chunks)
- **Per query:** ~$0.005 (one Claude call for generation + one for critic, with router + embedding adding ~$0.001)
- **Evaluation run:** ~$0.30 (RAGAS makes multiple LLM calls per question per metric)

For a portfolio demo with moderate traffic, expect under $10/month total.


---

## Acknowledgments

Built with:

- [Amazon Bedrock](https://aws.amazon.com/bedrock/) (Claude Haiku 4.5 + Titan Embeddings v2)
- [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph) for orchestration
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [RAGAS](https://docs.ragas.io/) for evaluation
- [Streamlit](https://streamlit.io/) for the UI
- Official [AWS Documentation](https://docs.aws.amazon.com/) as the knowledge base

---
