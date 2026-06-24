import os
from ollama import Client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# OLLAMA CONFIGURATION
# ==========================================
ollama_client = Client(host="http://localhost:11434")
OLLAMA_MODEL = "minimax-m3:cloud"

# ==========================================
# HARDCODED 5-YEAR FILE PATHS (Auto-Resolved)
# ==========================================
# This automatically finds the folder where this script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FILES = [
    os.path.join(SCRIPT_DIR, "data", "2021 10K.pdf"),
    os.path.join(SCRIPT_DIR, "data", "2022 10K.pdf"),
    os.path.join(SCRIPT_DIR, "data", "2023 10K.pdf"),
    os.path.join(SCRIPT_DIR, "data", "2024 10K.pdf"),
    os.path.join(SCRIPT_DIR, "data", "2025 10K.pdf")
]

# ==========================================
# ALL 20 HARDCODED EVALUATION QUESTIONS
# ==========================================
EVAL_QUESTIONS = [
    # --- DIRECT QUESTIONS ---
    {"id": "Q1", "category": "Direct", "text": "What was Domino's total revenue in the year 2024?"},
    {"id": "Q2", "category": "Direct", "text": "Who was the CEO of Domino's in 2023?"},
    {"id": "Q3", "category": "Direct", "text": "How many total supply chain centers did Domino's have in 2024?"},
    {"id": "Q4", "category": "Direct", "text": "What is the main address of Domino's corporate headquarters listed in the reports?"},
    {"id": "Q5", "category": "Direct", "text": "Did Domino's open more net stores internationally or in the US in 2024?"},
    
    # --- COMPARISON QUESTIONS ---
    {"id": "Q6", "category": "Comparison", "text": "Was total revenue higher in 2022 or in 2024?"},
    {"id": "Q7", "category": "Comparison", "text": "Did the number of U.S. company-owned stores increase or decrease from 2021 to 2023?"},
    {"id": "Q8", "category": "Comparison", "text": "Compare the net income of Domino's between 2023 and 2024—did it go up or down?"},
    {"id": "Q9", "category": "Comparison", "text": "Look at the risk factors in 2022 and 2024: did inflation or labor shortages become a bigger mention?"},
    {"id": "Q10", "category": "Comparison", "text": "Did Domino's spend more money on share repurchases in 2023 or 2024?"},
    
    # --- CALCULATION QUESTIONS ---
    {"id": "Q11", "category": "Calculation", "text": "Using the 2024 numbers subtract Capital Expenditures from Operating Cash Flow to find Free Cash Flow."},
    {"id": "Q12", "category": "Calculation", "text": "Divide the 2024 Operating Income by the 2024 Total Revenue to find the Operating Margin percentage."},
    {"id": "Q13", "category": "Calculation", "text": "What is the total sum of U.S. Franchise Revenues and International Franchise Revenues added together for 2024?"},
    {"id": "Q14", "category": "Calculation", "text": "Calculate the percentage change in total revenue from 2022 to 2024."},
    {"id": "Q15", "category": "Calculation", "text": "What is the difference in dollars between the advertising funds collected in 2023 versus 2024?"},
    
    # --- GENERAL QUESTIONS ---
    {"id": "Q16", "category": "General", "text": "What does management say is their main plan to grow store counts in the future?"},
    {"id": "Q17", "category": "General", "text": "What are the main risks listed regarding third-party delivery apps (like UberEats or DoorDash)?"},
    {"id": "Q18", "category": "General", "text": "How does the report explain the way Domino's makes money from its franchise stores?"},
    {"id": "Q19", "category": "General", "text": "What major supply chain challenges did the company highlight during the post-pandemic years?"},
    {"id": "Q20", "category": "General", "text": "Briefly summarize how weather or seasonal factors affect Domino's business according to the reports?"}
]

# ==========================================
# STEP 1: LOAD DOCUMENTS INDIVIDUALLY
# ==========================================
documents = []
print("Starting individual file extraction...")

for file_path in PDF_FILES:
    if os.path.exists(file_path):
        print(f"Loading document: {file_path}")
        file_loader = PyPDFLoader(file_path)
        # Pull out individual pages and extend our main repository array
        documents.extend(file_loader.load())
    else:
        print(f"Warning: File missing, skipping -> {file_path}")

print(f"Total cumulative pages safely loaded across corpus: {len(documents)}")

# ==========================================
# STEP 2: CHUNKING STRATEGY (V1.3 CLASS SPEC)
# ==========================================
# Baseline parameters: size 1000 characters, overlap window of 200
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Total vector chunks generated: {len(chunks)}")

# ==========================================
# STEP 3: EMBEDDING CONFIGURATION
# ==========================================
print("Computing vector metrics using all-MiniLM-L6-v2...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# STEP 4: VECTOR DATABASE STORE (FAISS)
# ==========================================
print("Populating local FAISS structure index...")
vector_db = FAISS.from_documents(chunks, embeddings)

# ==========================================
# STEP 5: AUTOMATED EVALUATION LOOP
# ==========================================
print("\n" + "="*60)
print(" RUNNING EVALUATION SUITE: VERSION 1.3 ")
print("="*60 + "\n")

for item in EVAL_QUESTIONS:
    print(f"Executing: {item['id']} ({item['category']})")
    
    # Retrieve top 3 relevant chunks via Cosine Similarity
    retrieved_docs = vector_db.similarity_search(item["text"], k=3)
    
    # Bundle retrieved context chunks cleanly
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Strict context constraint instruction prevents the LLM from using external data
    prompt = f"""
You are a strict financial assistant. You must answer the question using ONLY the provided text snippets. 
If the provided context does not contain the answer or the metrics necessary to perform the math, say exactly "Data Not Found". Do not make up facts.

Context:
{context}

Question:
{item['text']}

Answer:
"""
    
    # Execute request using local LLM instance
    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Display the results clearly for manual entry into your evaluation sheet
    print(f"\n--- RAG ANSWER [V1.3] FOR {item['id']} ---")
    print(response["message"]["content"].strip())
    print("-" * 60 + "\n")

print("Evaluation run completed successfully.")