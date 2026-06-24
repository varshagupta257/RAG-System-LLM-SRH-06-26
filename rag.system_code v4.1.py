import os
import shutil
import re
from ollama import Client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# OLLAMA CONFIGURATION
# ==========================================
ollama_client = Client(host="http://localhost:11434")
OLLAMA_MODEL = "minimax-m3:cloud"

# ==========================================
# PATH MANAGEMENT
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FILES = [
    os.path.join(SCRIPT_DIR, "data", "2021 10K.pdf"),
    os.path.join(SCRIPT_DIR, "data", "2022 10K.pdf"),
    os.path.join(SCRIPT_DIR, "data", "2023 10K.pdf"),
    os.path.join(SCRIPT_DIR, "data", "2024 10K.pdf"),
    os.path.join(SCRIPT_DIR, "data", "2025 10K.pdf")
]

CHROMA_PERSIST_DIR = os.path.join(SCRIPT_DIR, "chroma_db_storage_v4")

if os.path.exists(CHROMA_PERSIST_DIR):
    print(f"Resetting database cache directory at {CHROMA_PERSIST_DIR}...")
    shutil.rmtree(CHROMA_PERSIST_DIR)

# ==========================================
# EVALUATION QUESTIONS
# ==========================================
EVAL_QUESTIONS = [
    {"id": "Q1", "category": "Direct", "text": "What was Domino's total revenue in the year 2024?"},
    {"id": "Q2", "category": "Direct", "text": "Who was the CEO of Domino's in 2023?"},
    {"id": "Q3", "category": "Direct", "text": "How many total supply chain centers did Domino's have in 2024?"},
    {"id": "Q4", "category": "Direct", "text": "What is the main address of Domino's corporate headquarters listed in the reports?"},
    {"id": "Q5", "category": "Direct", "text": "Did Domino's open more net stores internationally or in the US in 2024?"},
    {"id": "Q6", "category": "Comparison", "text": "Was total revenue higher in 2022 or in 2024?"},
    {"id": "Q7", "category": "Comparison", "text": "Did the number of U.S. company-owned stores increase or decrease from 2021 to 2023?"},
    {"id": "Q8", "category": "Comparison", "text": "Compare the net income of Domino's between 2023 and 2024—did it go up or down?"},
    {"id": "Q9", "category": "Comparison", "text": "Look at the risk factors in 2022 and 2024: did inflation or labor shortages become a bigger mention?"},
    {"id": "Q10", "category": "Comparison", "text": "Did Domino's spend more money on share repurchases in 2023 or 2024?"},
    {"id": "Q11", "category": "Calculation", "text": "Using the 2024 numbers subtract Capital Expenditures from Operating Cash Flow to find Free Cash Flow."},
    {"id": "Q12", "category": "Calculation", "text": "Divide the 2024 Operating Income by the 2024 Total Revenue to find the Operating Margin percentage."},
    {"id": "Q13", "category": "Calculation", "text": "What is the total sum of U.S. Franchise Revenues and International Franchise Revenues added together for 2024?"},
    {"id": "Q14", "category": "Calculation", "text": "Calculate the percentage change in total revenue from 2022 to 2024."},
    {"id": "Q15", "category": "Calculation", "text": "What is the difference in dollars between the advertising funds collected in 2023 versus 2024?"},
    {"id": "Q16", "category": "General", "text": "What does management say is their main plan to grow store counts in the future?"},
    {"id": "Q17", "category": "General", "text": "What are the main risks listed regarding third-party delivery apps (like UberEats or DoorDash)?"},
    {"id": "Q18", "category": "General", "text": "How does the report explain the way Domino's makes money from its franchise stores?"},
    {"id": "Q19", "category": "General", "text": "What major supply chain challenges did the company highlight during the post-pandemic years?"},
    {"id": "Q20", "category": "General", "text": "Briefly summarize how weather or seasonal factors affect Domino's business according to the reports?"}
]

# ==========================================
# STEP 1: LOADING & METADATA ENRICHMENT
# ==========================================
processed_chunks = []
print("Loading documents and extracting Year tags...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)

for file_path in PDF_FILES:
    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        print(f"Processing structural layout metrics for: {filename}")
        
        # Pull out the specific 4 digit target year from the filename
        year_match = re.search(r"20\d{2}", filename)
        fiscal_year = int(year_match.group(0)) if year_match else 0
        
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        file_chunks = text_splitter.split_documents(pages)
        
        # Inject metadata parameters directly into the dictionaries
        for chunk in file_chunks:
            chunk.metadata["fiscal_year"] = fiscal_year
            processed_chunks.append(chunk)
    else:
        print(f"Warning: Missing required document input -> {file_path}")

print(f"Total structured metadata-tagged chunks loaded: {len(processed_chunks)}")

# ==========================================
# STEP 2: EMBEDDING & INDICES COMPILATION
# ==========================================
print("Generating textual vector representations using sentence-transformers...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building target vector persistent relational matrix tables...")
vector_store = Chroma.from_documents(
    documents=processed_chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)

# ==========================================
# STEP 3: REGEX YEAR DETECTOR
# ==========================================
def extract_year_filter(question_text):
    """Scans the text for years. Returns a filter dict if EXACTLY ONE year is found."""
    years_found = [int(y) for y in re.findall(r"\b202[1-5]\b", question_text)]
    years_found = list(set(years_found))
    
    if len(years_found) == 1:
        print(f"   [Metadata Guard]: Filter set to year {years_found[0]}")
        return {"fiscal_year": years_found[0]}
    return None

# ==========================================
# STEP 4: EVALUATION LOOP WITH SIMILARITY SEARCH
# ==========================================
print("\n" + "="*60)
print(" RUNNING EVALUATION SUITE: VERSION 4.0 (METADATA + SIMILARITY) ")
print("="*60 + "\n")

for item in EVAL_QUESTIONS:
    print(f"Executing: {item['id']} ({item['category']}) - {item['text']}")
    
    metadata_filter = extract_year_filter(item["text"])
    
    # Configure baseline similarity retrieval with a solid top-k depth
    search_kwargs = {"k": 10}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    # Run structural pipeline search 
    retrieved_docs = retriever.invoke(item["text"])
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Final production generation prompt
    prompt_template = f"""
You are a highly analytical corporate financial auditor. Review the context sections below to solve the user prompt. 

CRITICAL CONSTRAINTS:
1. Provide accurate numerical steps for all calculations using the values given in the text.
2. If the context does not explicitly state the specific metric or number required to execute a calculation or answer the question, output exactly "Data Not Found". Do not use outside knowledge.

Context Snippets:
{context_text}

Question:
{item['text']}

Answer:
"""
    
    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt_template}]
    )
    
    print(f"\n--- RAG ANSWER [V4.0] FOR {item['id']} ---")
    print(response["message"]["content"].strip())
    print("-" * 60 + "\n")

print("Evaluation run completed successfully.")