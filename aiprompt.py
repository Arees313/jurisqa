from flask import Flask, request, jsonify, send_from_directory
import json
import faiss
import numpy as np
import unicodedata
import re
from sentence_transformers import SentenceTransformer
import os
from werkzeug.middleware.proxy_fix import ProxyFix

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDINGS_FILE = "qa_embeddings.npy"
FAISS_INDEX_FILE = "qa.index"
DATA_FILE = "qa_data.json"
DISTANCE_THRESHOLD = 1.2
TOP_K = 3

# -------------------------------
# INITIALIZATION
# -------------------------------
app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# -------------------------------
# HELPER FUNCTIONS (same as before)
# -------------------------------
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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

def is_greeting(text):
    greetings = ["hello", "hi", "salaam", "hey", "peace", "assalamu alaikum"]
    return normalize(text) in greetings

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
    
    if is_greeting(question):
        response = {
            "answer": "Hello! Please ask a jurisprudential question.",
            "matched_question": None,
            "source": "N/A"
        }
    else:
        response = get_answer(question)
    
    print(f"📤 Returning response: {response}")
    return jsonify(response)

# CORS handling
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    # For Vercel deployment
    handler = app
    print("\n✅ Server ready! Access the UI at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)