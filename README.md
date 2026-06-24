# RAG-System-LLM-SRH-06-26
This repository is created for Assignment 3 - RAG system. 

# Requirements
Build a working Retrieval-Augmented Generation system on a document corpus of your choice (company annual reports, academic papers, technical documentation, legal texts, or any domain of professional interest). Evaluate it rigorously against defined questions and document every engineering decision you make.

# Step-by-step research tasks
1.  Choose a document corpus of at least 5 documents (approximately 50 pages total). Write 20 test questions spanning factual retrieval, multi-hop reasoning across documents, and synthesis tasks that require combining information from multiple sources.
2.	Implement a baseline RAG pipeline using LangChain or LlamaIndex: document loading, chunking, embedding model selection, vector store (FAISS or Chroma), retrieval, and generation. Document every parameter choice and why you made it.
3.	Experiment with at least two chunking strategies (e.g. fixed-size vs semantic chunking) and two retrieval configurations (e.g. different top-k values or MMR vs cosine similarity). Score answer quality on your 20 questions for each configuration.
4.	Research and implement one advanced improvement technique: re-ranking with a cross-encoder, Hypothetical Document Embeddings (HyDE), query expansion, or metadata filtering. Measure the quality delta compared to your baseline.
5.	Write a 1,500-word engineering report: decisions made and why, what worked well, failure analysis with specific examples, and what you would do differently when scaling to 10,000 documents in production.
