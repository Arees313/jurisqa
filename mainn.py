from flask import Flask, request, jsonify, send_from_directory
import json
import faiss
import numpy as np
import unicodedata
import re
from sentence_transformers import SentenceTransformer
import os
from werkzeug.middleware.proxy_fix import ProxyFix
import ollama
from functools import lru_cache
import time
from pyngrok import ngrok
# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "qa_embeddings.npy"
FAISS_INDEX_FILE = "qa.index"
DATA_FILE = "rag_knowledge_base.json"
DISTANCE_THRESHOLD = 1.4
TOP_K = 2
OLLAMA_MODEL = "" #llama3:8b-instruct-q4_0

# -------------------------------
# INITIALIZATION
# -------------------------------
app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Single space replacement
    return text

print("📖 Loading Q&A data...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

indexed_texts = []
for item in qa_data:
    text = item.get("text", "")
    indexed_texts.append(normalize(text))

model = SentenceTransformer(MODEL_NAME)

if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
    print("🔁 Loading existing index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    print("⚙️ Building new index...")
    embeddings = model.encode(indexed_texts, show_progress_bar=True, batch_size=16, device='cpu')
    np.save(EMBEDDINGS_FILE, embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("✅ Index built and saved")

# -------------------------------
# RAG LOGIC
# -------------------------------
@lru_cache(maxsize=100)
def generate_response_rag(user_query):
    start_time = time.time()
    
    query_embedding = model.encode([normalize(user_query)])
    distances, indices = index.search(query_embedding, TOP_K)

    # Check for very close matches first
    for i in range(TOP_K):
        if distances[0][i] <= 1.0 and indices[0][i] < len(qa_data):
            text = qa_data[indices[0][i]].get("text", "")
            source = qa_data[indices[0][i]].get("source", "")
            print(f"⏱️ Direct match time: {time.time()-start_time:.2f}s")
            return {
                "matched_question": "Direct match",
                "answer": text,
                "source": source,
                "is_refined": False
            }

    # Proceed with RAG for less perfect matches
    retrieved_contexts = []
    for i in range(TOP_K):
        if distances[0][i] <= DISTANCE_THRESHOLD and indices[0][i] < len(qa_data):
            text = qa_data[indices[0][i]].get("text", "")
            source = qa_data[indices[0][i]].get("source", "")
            retrieved_contexts.append(f"{text}\n(Source: {source})")

    if not retrieved_contexts:
        return {
            "matched_question": None,
            "answer": "Sorry, I couldn't find a relevant answer. Please contact the Jurisprudential Committee at alljnahalfkheah@yahoo.com.",
            "source": "No match found",
            "is_refined": True
        }

    full_context = "\n\n".join(retrieved_contexts)
    prompt = f"""You are a knowledgeable assistant on Islamic jurisprudence.
Below are relevant answers from authoritative sources:

{full_context}

Question: {user_query}

Answer directly and precisely using ONLY the provided context. Quote answers from the context and do not make up words. Include any relevant links exactly as provided. If the context contains a direct answer, use it verbatim when possible.

Answer:
"""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={'timeout': 10}  # 10 second timeout
        )
        print(f"⏱️ Total processing time: {time.time()-start_time:.2f}s")
        return {
            "matched_question": "RAG-based multi-match",
            "answer": response["message"]["content"].strip(),
            "source": "RAG using top similar questions",
            "is_refined": True
        }
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        # Fallback to returning the best match directly
        best_match_idx = indices[0][0]
        if distances[0][0] <= DISTANCE_THRESHOLD and best_match_idx < len(qa_data):
            text = qa_data[best_match_idx].get("text", "")
            source = qa_data[best_match_idx].get("source", "")
            return {
                "matched_question": "Best direct match",
                "answer": text,
                "source": source,
                "is_refined": False
            }

    return {
        "matched_question": None,
        "answer": "There was a problem generating the answer. Please try again later.",
        "source": "Generation failed",
        "is_refined": False
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
    try:
        ollama.list()
        print("✅ Ollama connected successfully")
    except Exception as e:
        print(f"⚠️ Ollama connection error: {e}")
        print("⚠️ Proceeding without Ollama refinement")

    # Start Ngrok tunnel
    public_url = ngrok.connect(8080).public_url
    print(f"🌐 Ngrok URL: {public_url} (May take a few seconds to become active)")
    print("🚀 Starting server... (Press Ctrl+C to stop)")
    
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        ngrok.kill()  # Close Ngrok tunnel when server stops