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

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDINGS_FILE = "qa_embeddings.npy"
FAISS_INDEX_FILE = "qa.index"
DATA_FILE = "qa_data.json"
DISTANCE_THRESHOLD = 1.2
TOP_K = 3
OLLAMA_MODEL = "mistral"

# -------------------------------
# INITIALIZATION
# -------------------------------
app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("📖 Loading Q&A data...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Replace the current text generation with this:
indexed_texts = []
for item in qa_data:
    # Boost keywords and tags with repetition
    keywords = " ".join([f"{kw} " * 3 for kw in item.get("keywords", [])])  # 3x weight
    tags = " ".join([f"{tag} " * 2 for tag in item.get("tags", [])])  # 2x weight
    category = item.get("category", "") + " " * 2  # 2x weight
    question = item.get("question", "")
    answer = item.get("answer", "")
    
    # Create weighted text
    text = f"{keywords}{tags}{category}Question: {question} Answer: {answer}"
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
    """Enhanced greeting detection with fuzzy matching"""
    text = normalize(text)
    greetings = ["hello", "hi", "salaam", "hey", "peace", "assalamu alaikum", 
                 "assalam", "salam", "marhaba", "ahlan", "greetings", "good morning",
                 "good evening", "good afternoon"]
    
    # Check for common greeting patterns
    if any(greet in text for greet in greetings):
        return True
    
    # Check for common misspellings
    greeting_patterns = [
        r"\b(?:s+a+l+a+m+|a+s+a+l+a+m+|s+a+l+a+a+m)\b",
        r"\b(?:h+i+|h+e+y+|h+a+y+)\b",
        r"\b(?:m+a+r+h+a+b+a)\b",
        r"\b(?:a+h+l+a+n)\b",
        r"\b(?:g+d\s*(?:morning|evening|afternoon))\b"
    ]
    
    return any(re.search(pattern, text) for pattern in greeting_patterns)

def refine_query_with_ollama(user_query, context):
    """
    Improved query refinement with better error handling
    """
    prompt = f"""
    You are helping interpret jurisprudential questions for an Islamic Q&A system. 
    
    Original question: '{user_query}'
    
    Available question context:
    {context}
    
    Your task is to:
    1. Understand the original question's intent
    2. Match it to the closest jurisprudential concept in our database
    3. Return ONLY a refined question that would match our content
    
    Rules:
    - Keep the refined question short and clear
    - Preserve the original meaning
    - Focus on jurisprudential aspects
    - Don't add new concepts not in the original
    - Return only the refined question, no commentary
    """
    
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={'temperature': 0.2, 'top_p': 0.9}
        )
        refined_query = response['response'].strip().strip('"').strip("'")
        
        # Basic validation
        if not refined_query or len(refined_query.split()) > 10:
            return user_query
            
        # Clean and return
        return re.sub(r'[^\w\s]', '', refined_query)
        
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return user_query  # Fallback to original query

def get_answer(user_query, top_k=TOP_K, distance_threshold=DISTANCE_THRESHOLD):
    # Removed automatic greeting shortcut
    query_embedding = model.encode([normalize(user_query)])
    distances, indices = index.search(query_embedding, top_k)

    # Get initial matches for context
    context_matches = []
    for i in range(top_k):
        if indices[0][i] < len(qa_data):
            context_matches.append(qa_data[indices[0][i]]["question"])
    context = "\n".join(context_matches)

    # First try with original query
    for i in range(top_k):
        if distances[0][i] <= distance_threshold and indices[0][i] < len(qa_data):
            matched_item = qa_data[indices[0][i]]
            similarity_score = 1 - (distances[0][i] / distance_threshold)
            if similarity_score > 0.7:  # Only return if highly confident
                return {
                    "matched_question": matched_item["question"],
                    "answer": matched_item["answer"],
                    "source": matched_item.get("source", "No source provided"),
                    "is_refined": False
                }

    # Try refinement if no good matches
    refined_query = refine_query_with_ollama(user_query, context)
    if refined_query != user_query:
        print(f"🔍 Refined query with Ollama: '{user_query}' -> '{refined_query}'")
        refined_embedding = model.encode([normalize(refined_query)])
        refined_distances, refined_indices = index.search(refined_embedding, top_k)
        
        for i in range(top_k):
            if refined_distances[0][i] <= distance_threshold and refined_indices[0][i] < len(qa_data):
                matched_item = qa_data[refined_indices[0][i]]
                return {
                    "matched_question": matched_item["question"],
                    "answer": matched_item["answer"],
                    "source": matched_item.get("source", "No source provided"),
                    "is_refined": True
                }

    # ====== STEP 3: KEYWORD FALLBACK MATCHING ======
    user_keywords = set(normalize(user_query).split())
    best_keyword_match = None
    best_keyword_score = 0
    
    for item in qa_data:
        item_keywords = set(
            [kw.lower() for kw in item.get("keywords", [])] +
            [tag.lower() for tag in item.get("tags", [])]
        )
        # Calculate intersection score (number of matching terms)
        match_score = len(user_keywords & item_keywords)
        
        # Only consider matches with at least 2 keyword overlaps
        if match_score >= 2 and match_score > best_keyword_score:
            best_keyword_score = match_score
            best_keyword_match = item
    
    if best_keyword_match:
        return {
            "matched_question": best_keyword_match["question"],
            "answer": f"Based on keyword matching: {best_keyword_match['answer']}",
            "source": best_keyword_match.get("source", "No source provided"),
            "is_refined": False
        }
    # ====== END OF STEP 3 ======

    # Final fallback - try to find the closest match even if above threshold
    best_match_idx = np.argmin(distances[0])
    if best_match_idx < len(qa_data):
        matched_item = qa_data[indices[0][best_match_idx]]
        return {
            "matched_question": matched_item["question"],
            "answer": f"I found this potentially relevant information: {matched_item['answer']} \n\nIf this doesn't answer your question, please try rephrasing or contact the Jurisprudential Committee at alljnahalfkheah@yahoo.com.",
            "source": matched_item.get("source", "No source provided"),
            "is_refined": False
        }

    return {
        "matched_question": None,
        "answer": "I couldn't find a precise answer to your question. For specific jurisprudential matters, please contact the Jurisprudential Committee at alljnahalfkheah@yahoo.com.",
        "source": "N/A",
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
    response = get_answer(question)
    print(f"📤 Returning response: {response}")
    return jsonify(response)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    try:
        ollama.list()
        print("✅ Ollama connected successfully")
    except Exception as e:
        print(f"⚠️ Ollama connection error: {e}")
        print("⚠️ Proceeding without Ollama refinement")
    
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))