# RAG-System-LLM-SRH-06-26
- This repository is created for Assignment 3 - RAG system.
- Subject: LLM, Prompting and Agentic AI
- Course and University: M.Sc. AI in Business, SRH Berlin university of Applied Science
= Semester: Summer Semester, 2026

# Requirements
Build a working Retrieval-Augmented Generation system on a document corpus of your choice (company annual reports, academic papers, technical documentation, legal texts, or any domain of professional interest). Evaluate it rigorously against defined questions and document every engineering decision you make.

# Step-by-step research tasks
- 1.  Choose a document corpus of at least 5 documents (approximately 50 pages total). Write 20 test questions spanning factual retrieval, multi-hop reasoning across documents, and synthesis tasks that require combining information from multiple sources.
- 2.	Implement a baseline RAG pipeline using LangChain or LlamaIndex: document loading, chunking, embedding model selection, vector store (FAISS or Chroma), retrieval, and generation. Document every parameter choice and why you made it.
- 3.	Experiment with at least two chunking strategies (e.g. fixed-size vs semantic chunking) and two retrieval configurations (e.g. different top-k values or MMR vs cosine similarity). Score answer quality on your 20 questions for each configuration.
- 4.	Research and implement one advanced improvement technique: re-ranking with a cross-encoder, Hypothetical Document Embeddings (HyDE), query expansion, or metadata filtering. Measure the quality delta compared to your baseline.
- 5.	Write a 1,500-word engineering report: decisions made and why, what worked well, failure analysis with specific examples, and what you would do differently when scaling to 10,000 documents in production.

# Introduction
This project evaluates a Retrieval-Augmented Generation (RAG) pipeline designed to extract and synthesize financial insights from five consecutive years of Domino’s Pizza 10-K reports (2021–2025).

# Methodology and Evaluation
To ensure a rigorous assessment, I developed a benchmark suite of 20 evaluation questions categorized into four distinct levels of complexity:
- Direct: Fact-based extraction (e.g., revenue, CEO, store counts).
- Comparison: Longitudinal analysis (e.g., year-over-year revenue trends).
- Calculation: Quantitative reasoning requiring multi-step arithmetic (e.g., deriving Free Cash Flow).
- General: Contextual and qualitative summarization (e.g., risk factors and growth strategies).

The baseline setup utilized local embeddings and initial retrieval strategies; however, as the project evolved, I systematically upgraded the architecture. By refining chunking strategies, migrating to persistent databases like ChromaDB, and implementing sophisticated Metadata Guards and temporal offsets, the system’s ability to resolve complex financial queries improved significantly.

As the iterations progressed from version 1.1 to 4.2, the pipeline’s performance path became clear: transitioning from basic semantic similarity—which often suffered from cross-year data contamination—to a deterministic, filtered retrieval model that mirrors the precision required for professional corporate auditing.
