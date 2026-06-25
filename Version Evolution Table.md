# Version	Configuration / Change	Purpose
- V1.1	Minimax, Chunk: 500/50, $k=3$	Establishing a baseline retrieval environment.
- V1.2	Switched to Gemini API	Testing alternative LLM performance for reasoning but hit the rate limits.
- V1.3	Reverted to Minimax, Chunk: 1000/200	Resolved API rate limits; increased context per chunk.
- V2.1	Switched to ChromaDB	Added persistent storage to prevent re-indexing loss.
- V2.2	Chunk: 3000/500, k=10	Captured larger data blocks; increased retrieval "depth."
- V3.1	Added MMR Retrieval	Optimized to pull diverse, non-repetitive snippets.
- V3.2	Lambda \lambda = 0.75	Fine-tuned the balance between relevance and diversity.
- V4.1	Metadata & Regex filtering	Forced the system to look only at specified document years.
- V4.2	+1 Year Temporal Offset	Aligned fiscal data with filing dates to stop cross-year errors.

# Technical Analysis of Improvements
- The Baseline Struggle (V1.1–V2.2): Initially, the system suffered from "context dilution." Using small chunks (500) meant the model didn't have enough data to perform calculations. Increasing the chunk size to 3000 (V2.2) provided the model with enough text to see full financial tables, but the retrieval was still "blind" to the year.
- The Relevance Phase (V3.1–V3.2): I introduced Maximal Marginal Relevance (MMR) to prevent the system from returning five identical paragraphs from the same section. Adjusting  to 0.75 provided the "sweet spot" where the model retrieved broad context without losing focus on the specific question.
- The Precision Breakthrough (V4.1–V4.2): This was the most critical evolution. Even with better retrieval, the LLM hallucinated years. By using regex to extract year metadata and applying a +1 Filing Offset, I created a deterministic barrier. The system no longer guessed which year the document belonged to; it knew based on the file's metadata.

