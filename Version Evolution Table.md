# Version,Strategy,Core Logic,Key Technical Milestone,Resulting Benefit
- V1.1,Naive RAG,FAISS Similarity,Baseline implementation,Proof of concept.
- V1.2,Naive RAG,Gemini Flash,Cloud-based LLM integration,Verified remote inference.
- V1.3,Naive RAG,Expanded Context,"1,000 token chunk size",Improved context retention.
- V2.1,Metadata-Lite,Filename Parsing,Automated year-tagging logic,Initial data lineage tracking.
- V2.2,High-Density,Optimized Chunks,"3,000/500 split-overlap",High-density information recall.
- V3.1,Advanced,Relevance Tuning,Optimized search depth,Improved signal-to-noise ratio.
- V3.2,Advanced,λ=0.75 weighting,Balanced relevance/diversity,Optimized context density.
- V4.1,Deterministic,Regex Query Parsing,Dynamic temporal extraction,Enabled natural language temporal queries.
- V4.2,Production,+1 Offset Logic,Fiscal-to-Filing Mapping,"Eliminated ""lag"" errors in financial reporting."

# Evolution using distinct phases:
- Phase 1: Retrieval Baseline (V1.x): Establishing the foundation of PDF parsing and basic vector search.
- Phase 2: Persistence & Scale (V2.x): Moving from ephemeral memory to permanent, high-density vector databases.
- Phase 3: Cognitive Search (V3.x): Introducing "Diversity" to retrieval, moving away from redundant chunks to a multi-faceted information set.
- Phase 4: Deterministic Logic (V4.x): The "Auditor" phase, where semantic search is constrained by deterministic filters and temporal offsets to eliminate the risk of cross-year financial data mixing.
