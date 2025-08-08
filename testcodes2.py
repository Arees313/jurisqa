from flask import Flask, request, jsonify, send_from_directory
import json
import faiss
import numpy as np
import unicodedata
import re
from sentence_transformers import SentenceTransformer
import os
from werkzeug.middleware.proxy_fix import ProxyFix
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "qa_embeddings.npy"
FAISS_INDEX_FILE = "qa.index"
DATA_FILE = "rag_knowledge_base.json"
DISTANCE_THRESHOLD = 1.0  # More strict threshold
SUGGESTION_THRESHOLD = 1.3  # Range for "did you mean" suggestions
TOP_K = 3  # Get more results for better suggestions

# -------------------------------
# INITIALIZATION WITH AUTO-RELOAD
# -------------------------------
app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Single space replacement
    return text

def extract_question_from_text(text):
    """Extract the main question from a Q&A text"""
    if "Question:" in text and "Answer:" in text:
        question_part = text.split("Answer:")[0].replace("Question:", "").strip()
        # Clean up common question formatting
        question_part = re.sub(r'^(What is the ruling on|Question \d+:|Q\d+/)', '', question_part).strip()
        # Limit length for readability
        if len(question_part) > 120:
            question_part = question_part[:120] + "..."
        return question_part
    return text[:100] + "..." if len(text) > 100 else text

# Global variables for data storage
qa_data = []
indexed_texts = []
index = None
model = SentenceTransformer(MODEL_NAME)

def load_data(force_rebuild=False):
    global qa_data, indexed_texts, index

    print("\n📖 Loading Q&A data...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    indexed_texts = [normalize(item.get("text", "")) for item in qa_data]

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
    print("✅ Knowledge base loaded!")

# Initial data load
load_data()

# File watcher setup
if not os.environ.get("WERKZEUG_RUN_MAIN"):  # Prevent duplicate in Flask reloader
    class JsonUpdateHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.src_path.endswith(DATA_FILE):
                print("\n🔄 Detected knowledge base change - reloading with index rebuild...")
                try:
                    load_data(force_rebuild=True)
                except Exception as e:
                    print(f"⚠️ Failed to reload: {str(e)}")

    observer = Observer()
    observer.schedule(JsonUpdateHandler(), path='.', recursive=False)
    observer.start()
    print("🔍 File watcher started for auto-reload")

# -------------------------------
# IMPROVED RAG LOGIC
# -------------------------------
def generate_response_rag(user_query):
    start_time = time.time()
    
    query_embedding = model.encode([normalize(user_query)])
    distances, indices = index.search(query_embedding, TOP_K)

    # Log the matching scores for debugging
    print(f"🔍 Search results for '{user_query}':")
    for i in range(TOP_K):
        if indices[0][i] < len(qa_data):
            score = float(distances[0][i])  # Convert to Python float for consistency
            preview = qa_data[indices[0][i]].get("text", "")[:100] + "..."
            print(f"  Rank {i+1}: Score={score:.3f} - {preview}")

    # Check for very close matches first (high confidence)
    for i in range(TOP_K):
        if distances[0][i] <= DISTANCE_THRESHOLD and indices[0][i] < len(qa_data):
            text = qa_data[indices[0][i]].get("text", "")
            source = qa_data[indices[0][i]].get("source", "")
            print(f"⏱️ Direct match time: {time.time()-start_time:.2f}s (Score: {float(distances[0][i]):.3f})")
            return {
                "matched_question": "Direct match",
                "answer": text,
                "source": source,
                "is_refined": False,
                "confidence": "high"
            }

    # Check for suggestion matches (medium confidence)
    suggestions = []
    for i in range(TOP_K):
        if DISTANCE_THRESHOLD < distances[0][i] <= SUGGESTION_THRESHOLD and indices[0][i] < len(qa_data):
            text = qa_data[indices[0][i]].get("text", "")
            question_preview = extract_question_from_text(text)
            suggestions.append({
                "question": question_preview,
                "score": float(distances[0][i]),  # Convert numpy.float32 to Python float
                "full_text": text,
                "source": qa_data[indices[0][i]].get("source", "")
            })

    # If we have suggestions, offer them
    if suggestions:
        suggestions_text = "\n".join([f"• {s['question']}" for s in suggestions[:2]])  # Limit to 2 suggestions
        print(f"💡 Providing suggestions with scores: {[float(s['score']) for s in suggestions[:2]]}")
        return {
            "matched_question": None,
            "answer": f"Sorry, I couldn't find an exact match for your question. Did you mean one of these:\n\n{suggestions_text}\n\nIf none of these match your question, please contact the Jurisprudential Committee at alljnahalfkheah@yahoo.com for guidance.",
            "source": "Suggestions provided",
            "is_refined": True,
            "confidence": "medium",
            "suggestions": suggestions[:2]
        }

    # No good matches found
    print(f"❌ No matches found. Best score was {float(distances[0][0]):.3f}, suggestion threshold is {SUGGESTION_THRESHOLD}")
    return {
        "matched_question": None,
        "answer": "Sorry, I couldn't find a relevant answer to your question. Please contact the Jurisprudential Committee at alljnahalfkheah@yahoo.com for personalized guidance on Islamic jurisprudence matters.",
        "source": "No match found",
        "is_refined": True,
        "confidence": "low"
    }

# -------------------------------
# FLASK ROUTES
# -------------------------------
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    print(f"📩 Received question: '{question}'")
    response = generate_response_rag(question)
    print(f"📤 Returning response: {response}")
    return jsonify(response)

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
    print("🚀 Starting TEST Flask server on port 8081...")
    print("📍 Local access: http://localhost:8081")
    print("🧪 Testing improved matching logic")
    print("(Press Ctrl+C to stop)")
    
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=8081)
    except KeyboardInterrupt:
        print("\n🛑 Test server stopped")
        if 'observer' in globals():
            observer.stop()
            observer.join()
