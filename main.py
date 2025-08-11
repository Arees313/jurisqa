from flask import Flask, request, jsonify, send_from_directory
import json
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import os
from werkzeug.middleware.proxy_fix import ProxyFix
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

# ------------------------------
# CONFIGURATION
# -------------------------------
MODEL_NAME = "all-mpnet-base-v2"
EMBEDDINGS_FILE = "qa_embeddings.npy"
FAISS_INDEX_FILE = "qa.index"
DATA_FILE = "rag_knowledge_base.json"
DISTANCE_THRESHOLD = 1.0
SUGGESTION_THRESHOLD = 1.3
TOP_K = 5
HYBRID_ALPHA = 0.9

# -------------------------------
# INITIALIZATION
# -------------------------------
app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Globals
qa_data = []
indexed_texts = []
tokenized_corpus = []
index = None
bm25 = None
model = SentenceTransformer(MODEL_NAME)
observer = None  # file watcher

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_question_from_text(text):
    if "Question:" in text and "Answer:" in text:
        question_part = text.split("Answer:")[0].replace("Question:", "").strip()
        question_part = re.sub(r'^(What is the ruling on|Question \d+:|Q\d+/)', '', question_part).strip()
        if len(question_part) > 120:
            question_part = question_part[:120] + "..."
        return question_part
    return text[:100] + "..." if len(text) > 100 else text

def load_data(force_rebuild=False):
    """
    Loads Q&A data, rebuilds embeddings/index if necessary.
    """
    global qa_data, indexed_texts, tokenized_corpus, index, bm25

    print("\n📖 Loading Q&A data...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    qa_data = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data

    indexed_texts = []
    tokenized_corpus = []
    for item in qa_data:
        text_content = item.get("text", "")
        related_questions = item.get("related_questions", [])
        if related_questions:
            text_content += " " + " ".join(related_questions)
        normalized_text = normalize(text_content)
        indexed_texts.append(normalized_text)
        tokenized_corpus.append(normalized_text.split())

    bm25 = BM25Okapi(tokenized_corpus)

    # Decide whether to rebuild
    if force_rebuild or not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(EMBEDDINGS_FILE):
        print("⚙️ Building new FAISS index with fresh embeddings...")
        embeddings = model.encode(indexed_texts, show_progress_bar=True, batch_size=16, device='cpu')
        np.save(EMBEDDINGS_FILE, embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        faiss.write_index(index, FAISS_INDEX_FILE)
    else:
        print("🔁 Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_FILE)

    print(f"✅ Knowledge base loaded! ({len(qa_data)} entries)")

# -------------------------------
# FILE WATCHER FOR AUTO-RELOAD
# -------------------------------
class ReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(DATA_FILE):
            print("🔄 Data file changed, rebuilding index & embeddings...")
            try:
                load_data(force_rebuild=True)
            except Exception as e:
                print(f"❌ Failed to reload data: {e}")

def start_file_watcher():
    global observer
    event_handler = ReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print(f"👀 Watching {DATA_FILE} for changes...")

# -------------------------------
# HYBRID SEARCH
# -------------------------------
def hybrid_search(query: str, top_k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
    query_embedding = model.encode([normalize(query)])
    faiss_distances, faiss_indices = index.search(query_embedding, top_k * 2)
    tokenized_query = normalize(query).split()
    bm25_scores = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    normalized_bm25 = [score / max_bm25 for score in bm25_scores]
    combined_scores = []
    for idx in range(len(indexed_texts)):
        faiss_sim = 1 / (1 + faiss_distances[0][0]) if idx in faiss_indices[0] else 0
        combined = (HYBRID_ALPHA * faiss_sim) + ((1 - HYBRID_ALPHA) * normalized_bm25[idx])
        combined_scores.append((idx, combined))
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = np.array([idx for idx, _ in combined_scores[:top_k]])
    top_scores = np.array([1 - score for _, score in combined_scores[:top_k]])
    return top_scores.reshape(1, -1), top_indices.reshape(1, -1)

# -------------------------------
# RESPONSE GENERATION
# -------------------------------
def generate_response_rag(user_query):
    start_time = time.time()
    distances, indices = hybrid_search(user_query, TOP_K)

    print(f"🔍 Hybrid search results for '{user_query}':")
    for i in range(TOP_K):
        if indices[0][i] < len(qa_data):
            print(f"  Rank {i+1}: Score={float(distances[0][i]):.3f} | Match='{extract_question_from_text(qa_data[indices[0][i]].get('text', ''))}'")


    def get_related_questions(exclude_index=None):
        related = []
        for i in range(TOP_K):
            if indices[0][i] < len(qa_data) and indices[0][i] != exclude_index:
                text = qa_data[indices[0][i]].get("text", "")
                related.append({
                    "question": extract_question_from_text(text),
                    "score": float(distances[0][i]),
                    "full_text": text,
                    "source": qa_data[indices[0][i]].get("source", "")
                })
        return related[:2]

    for i in range(TOP_K):
        if distances[0][i] <= DISTANCE_THRESHOLD and indices[0][i] < len(qa_data):
            text = qa_data[indices[0][i]].get("text", "")
            source = qa_data[indices[0][i]].get("source", "")
            related_questions = get_related_questions(exclude_index=indices[0][i])
            if related_questions:
                text += (
                    "\n\n"
                    "<hr>"
                    "<b>You might also be interested in:</b><br>"
                    + "<br>".join([f"• {q['question']}" for q in related_questions])
                    + "<br><hr>"
                )

            print(f"⏱️ Direct match in {time.time()-start_time:.2f}s")
            return {
                "matched_question": "Direct match",
                "answer": text,
                "source": source,
                "is_refined": False,
                "confidence": "high",
                "related_questions": related_questions
            }

    suggestions = []
    for i in range(TOP_K):
        if DISTANCE_THRESHOLD < distances[0][i] <= SUGGESTION_THRESHOLD:
            text = qa_data[indices[0][i]].get("text", "")
            suggestions.append({
                "question": extract_question_from_text(text),
                "score": float(distances[0][i]),
                "full_text": text,
                "source": qa_data[indices[0][i]].get("source", "")
            })

    if suggestions:
        suggestions_text = "<br><hr>".join([f"• {s['question']}" for s in suggestions[:2]])
        return {
            "matched_question": None,
            "answer": f"Sorry, I couldn't find an exact match. Did you mean:\n\n{suggestions_text}",
            "source": "Suggestions provided",
            "is_refined": True,
            "confidence": "medium",
            "suggestions": suggestions[:2],
            "related_questions": get_related_questions()
        }

    return {
        "matched_question": None,
        "answer": "Sorry, I couldn't find a relevant answer. Please contact the Jurisprudential Committee at alljnahalfkheah@yahoo.com.",
        "source": "No match found",
        "is_refined": True,
        "confidence": "low",
        "related_questions": get_related_questions()
    }

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "knowledge_base_size": len(qa_data),
        "model_loaded": model is not None,
        "index_loaded": index is not None
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '')
        print(f"📩 Received: '{question}'")
        return jsonify(generate_response_rag(question))
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == '__main__':
    load_data(force_rebuild=True)  # ensure fresh start
    start_file_watcher()  # watch for changes
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        if observer:
            observer.stop()
            observer.join()
