from flask import Flask, request, jsonify, send_from_directory
import json
import os
from werkzeug.middleware.proxy_fix import ProxyFix
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import BaseEmbedding
import ollama
import numpy as np

app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

class CustomEmbedding(BaseEmbedding):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self._model = SentenceTransformer(model_name)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

print("📖 Loading Q&A data and building index...")
with open("qa_data.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Load the embedding model
embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Precompute embeddings for all questions in the dataset
qa_embeddings = [
    {
        "question": item["question"],
        "answer": item["answer"],
        "embedding": embedding_model.encode(
            f"{item['question']} {' '.join(item.get('tags', []))} {' '.join(item.get('keywords', []))}"
        )
    }
    for item in qa_data
]

documents = []
for item in qa_data:
    q = item["question"]
    a = item["answer"]
    full_text = f"Q: {q}\nA: {a}"
    documents.append(Document(text=full_text))

custom_embed_model = CustomEmbedding()
llm = Ollama(model="llama3")

Settings.llm = llm
Settings.embed_model = custom_embed_model
Settings.node_parser = SimpleNodeParser()

nodes = Settings.node_parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine()

def is_greeting(text):
    greetings = ["hello", "hi", "salaam", "hey", "peace",
                "assalamu alaikum", "assalam", "salam",
                "marhaba", "ahlan", "greetings"]
    text = text.lower()
    return any(greet in text for greet in greetings)

def normalize_text(text):
    """
    Normalize text by converting to lowercase and removing punctuation.
    """
    import string
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def refine_query(question):
    """
    Refine the user's question to improve matching.
    """
    prompt = f"""
    You are an jurisprudential assistant for Ahmed AlHasan and for the followers of Ahmed Alhasan. 
    Always try to quote the answer from the provided context.
    Never include an answer that is not in the qa_data.json file.
    Never include an answer from other sources.
    Refine the following question to make it concise and clear for matching with a knowledge base:
    Original: '{question}'
    Rules:
    - Keep the core meaning intact.
    - Use standard Islamic terminology.
    - Return only the refined question as plain text.
    """
    try:
        response = ollama.generate(
            model="llama3",
            prompt=prompt,
            options={'temperature': 0.0}
        )
        refined_question = response['response'].strip()
        print(f"📝 Refined question: '{refined_question}'")
        return refined_question
    except Exception as e:
        print(f"⚠️ Error refining query: {e}")
        return question  # Fallback to the original question if refinement fails

def get_answer_from_json(question):
    """
    Retrieve the most relevant answer from the JSON file based on the user's question.
    """
    question_embedding = embedding_model.encode(question)
    best_match = None
    highest_score = -1

    for item in qa_embeddings:
        # Compute cosine similarity
        similarity = np.dot(question_embedding, item["embedding"]) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(item["embedding"])
        )
        
        # Log the similarity comparison
        print(f"🔍 Comparing with: '{item['question']}' | Similarity: {similarity:.2f}")

        if similarity > highest_score:
            highest_score = similarity
            best_match = item

    # Check if the similarity score meets the threshold
    if highest_score >= 0.6:  # Adjust threshold as needed
        print(f"✅ Best match found with similarity {highest_score:.2f}: '{best_match['question']}'")
        return best_match["answer"]  # Return the exact answer from the JSON file
    else:
        print(f"❌ No relevant match found. Highest similarity: {highest_score:.2f}")
        return "Sorry, I don't have an answer for that. If you do not find your answer here, email The Jurisprudential Committee at: alljnahalfkheah@yahoo.com."
    
def validate_response(response, context):
    """
    Validate the AI's response to ensure it strictly adheres to the provided context.
    """
    if response not in context:
        print("⚠️ Response does not strictly adhere to the context. Returning fallback response.")
        return "Sorry, I don't have an answer for that. If you do not find your answer here, email The Jurisprudential Committee at: alljnahalfkheah@yahoo.com."
    return response

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '').strip()

    print(f"📩 Received question: '{question}'")

    if is_greeting(question):
        response = {
            "answer": "Assalam Aleikum! How can I assist with your jurisprudential questions today?",
            "source": "System",
            "is_refined": False
        }
    else:
        # Step 1: Refine the user's question
        refined_question = refine_query(question)

        # Step 2: Retrieve context using semantic search
        context = get_answer_from_json(refined_question)  # Updated function call

        if context.startswith("Sorry"):  # Check if no relevant match was found
            response = {
                "answer": context,
                "source": "Fallback",
                "is_refined": True
            }
        else:
            # Directly use the answer from the JSON file
            print(f"📝 Direct response from JSON: '{context}'")
            response = {
                "answer": context,
                "source": "Direct match",
                "is_refined": True
            }

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

    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))