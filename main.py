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
MODEL_NAME = "all-mpnet-base-v2"
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
        raw_data = json.load(f)
    
    # Convert to list format if it's in dict format
    if isinstance(raw_data, dict):
        qa_data = list(raw_data.values())
    else:
        qa_data = raw_data

    # Create enhanced indexed texts that include related questions
    indexed_texts = []
    for item in qa_data:
        # Start with the main text
        text_content = item.get("text", "")
        
        # Add related questions to improve search matching
        related_questions = item.get("related_questions", [])
        if related_questions:
            # Join related questions with the main text
            questions_text = " ".join(related_questions)
            text_content = f"{text_content} {questions_text}"
        
        indexed_texts.append(normalize(text_content))

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

    # Prepare related questions (always show closest matches excluding the main answer)
    def get_related_questions(exclude_index=None):
        related = []
        for i in range(TOP_K):
            if indices[0][i] < len(qa_data) and indices[0][i] != exclude_index:
                text = qa_data[indices[0][i]].get("text", "")
                question_preview = extract_question_from_text(text)
                related.append({
                    "question": question_preview,
                    "score": float(distances[0][i]),
                    "full_text": text,
                    "source": qa_data[indices[0][i]].get("source", "")
                })
        return related[:2]  # Return top 2 related questions

    # Check for very close matches first (high confidence)
    for i in range(TOP_K):
        if distances[0][i] <= DISTANCE_THRESHOLD and indices[0][i] < len(qa_data):
            text = qa_data[indices[0][i]].get("text", "")
            source = qa_data[indices[0][i]].get("source", "")
            related_questions = get_related_questions(exclude_index=indices[0][i])
            
            # Add related questions to the answer (moved lower with more spacing)
            if related_questions:
                related_text = "\n\n\n<b>You might also be interested in:</b>\n" + "\n".join([f"• {q['question']}" for q in related_questions])
                text += related_text
            
            print(f"⏱️ Direct match time: {time.time()-start_time:.2f}s (Score: {float(distances[0][i]):.3f})")
            return {
                "matched_question": "Direct match",
                "answer": text,
                "source": source,
                "is_refined": False,
                "confidence": "high",
                "related_questions": related_questions
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
        # Also get additional related questions
        all_related = get_related_questions()
        
        print(f"💡 Providing suggestions with scores: {[float(s['score']) for s in suggestions[:2]]}")
        return {
            "matched_question": None,
            "answer": f"Sorry, I couldn't find an exact match for your question. Did you mean one of these:\n\n{suggestions_text}\n\nIf none of these match your question, please contact the Jurisprudential Committee at alljnahalfkheah@yahoo.com for guidance.",
            "source": "Suggestions provided",
            "is_refined": True,
            "confidence": "medium",
            "suggestions": suggestions[:2],
            "related_questions": all_related
        }

    # No good matches found - still show related questions
    all_related = get_related_questions()
    no_match_answer = "Sorry, I couldn't find a relevant answer to your question. Please contact the Jurisprudential Committee at alljnahalfkheah@yahoo.com for personalized guidance on Islamic jurisprudence matters."
    
    if all_related:
        related_text = "\n\n\n<b>Here are some related topics that might help:</b>\n" + "\n".join([f"• {q['question']}" for q in all_related])
        no_match_answer += related_text
    
    print(f"❌ No matches found. Best score was {float(distances[0][0]):.3f}, suggestion threshold is {SUGGESTION_THRESHOLD}")
    return {
        "matched_question": None,
        "answer": no_match_answer,
        "source": "No match found",
        "is_refined": True,
        "confidence": "low",
        "related_questions": all_related
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
    request_start = time.time()
    try:
        data = request.get_json()
        question = data.get('question', '')
        print(f"📩 Received question: '{question}' at {time.strftime('%H:%M:%S')}")
        response = generate_response_rag(question)
        processing_time = time.time() - request_start
        print(f"📤 Returning response in {processing_time:.2f}s: {response.get('confidence', 'unknown')} confidence")
        return jsonify(response)
    except Exception as e:
        processing_time = time.time() - request_start
        print(f"❌ Request failed after {processing_time:.2f}s: {str(e)}")
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
    print("🚀 Starting Flask server on port 8080...")
    print("🌐 Access via: https://ahmed313.com")
    print("📍 Local access: http://localhost:8080")
    print("✨ Enhanced with improved semantic search")
    print("(Press Ctrl+C to stop)")
    
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        if 'observer' in globals():
            observer.stop()
            observer.join()