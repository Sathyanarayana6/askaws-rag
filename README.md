# AskAWS — Multi-Agent RAG Assistant for AWS Documentation

> A friendly assistant that helps you learn AWS. Four AI agents work together to answer your questions with grounded, cited responses from official AWS documentation.

**Live demo:** _coming soon (Phase 7 deployment)_
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