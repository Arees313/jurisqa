import json
import faiss
import numpy as np
import unicodedata
import re
from sentence_transformers import SentenceTransformer
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

# Moun# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDINGS_FILE = "qa_embeddings.npy"
FAISS_INDEX_FILE = "qa.index"
DATA_FILE = "qa_data.json"
DISTANCE_THRESHOLD = 1.2
TOP_K = 3

# -------------------------------
# NORMALIZE TEXT
# -------------------------------
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------
# LOAD DATA
# -------------------------------
print("📖 Loading Q&A data...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

indexed_texts = []
for item in qa_data:
    keywords = " ".join(item.get("keywords", []))
    tags = " ".join(item.get("tags", []))
    category = item.get("category", "")
    question = item.get("question", "")
    answer = item.get("answer", "")
    text = f"Question: {question}\nKeywords: {keywords}\nTags: {tags}\nCategory: {category}\nAnswer: {answer}"
    indexed_texts.append(normalize(text))

# -------------------------------
# LOAD EMBEDDINGS & INDEX
# -------------------------------
model = SentenceTransformer(MODEL_NAME)

if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
    print("🔁 Loading existing index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    print("⚙️ Building new index...")
    embeddings = model.encode(indexed_texts, show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("✅ Index built and saved")

# -------------------------------
# GREETING CHECK
# -------------------------------
def is_greeting(text):
    greetings = ["hello", "hi", "salaam", "hey", "peace", "assalamu alaikum"]
    return normalize(text) in greetings

# -------------------------------
# GET ANSWER FUNCTION
# -------------------------------
def get_answer(user_query, top_k=TOP_K, distance_threshold=DISTANCE_THRESHOLD):
    query_embedding = model.encode([normalize(user_query)])
    distances, indices = index.search(query_embedding, top_k)

    for i in range(top_k):
        best_distance = distances[0][i]
        best_idx = indices[0][i]
        if best_distance <= distance_threshold:
            matched_item = qa_data[best_idx]
            return {
                "matched_question": matched_item["question"],
                "answer": matched_item["answer"],
                "source": matched_item.get("source", "No source provided")
            }

    return {
        "matched_question": None,
        "answer": "Sorry, I couldn't find a close match. Try rephrasing your question.",
        "source": "N/A"
    }

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionInput(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: QuestionInput):
    print(f"📩 Received question: '{data.question}'")
    
    if is_greeting(data.question):
        response = {
            "answer": "Hello! Please ask a jurisprudential question.",
            "matched_question": None,
            "source": "N/A"
        }
    else:
        response = get_answer(data.question)
    
    print(f"📤 Returning response: {response}")
    return response

print("\n✅ Server ready! Access the UI at http://localhost:8000")