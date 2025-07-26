import json
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_index import VectorStoreIndex, Document, ServiceContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.base import BaseEmbedding
import re
from typing import List, Dict, Any, Optional
from pydantic import PrivateAttr

from sentence_transformers import SentenceTransformer
from llama_index.embeddings.base import BaseEmbedding

class CustomEmbedding(BaseEmbedding):
    def __init__(self):
        # Skip Pydantic's init completely
        object.__setattr__(self, '_model', SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()
    
    def _get_query_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    async def _aget_query_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)
    
    def get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self._model.encode(texts, **kwargs).tolist()
    
    def get_query_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.get_text_embedding_batch(texts, **kwargs)
# ... [rest of your code remains exactly the same] ...

# Constants
GREETINGS = [
    "hello", "hi", "hey", "salaam", "peace be upon you", 
    "who are you", "how are you", "greetings", "assalam", "as-salam",
    "السلام عليكم", "سلام", "مرحبا", "اهلا"
]

# Load Q&A data
def load_qa_data(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    qa_list = load_qa_data("qa_data.json")
    ALL_SOURCES = {item["source"] for item in qa_list}
except Exception as e:
    print(f"Error loading qa_data.json: {e}")
    exit(1)

# Enhanced keyword extraction
def extract_keywords(text: str) -> List[str]:
    """Universal keyword extraction for Islamic jurisprudence"""
    islamic_terms = re.findall(r'[\u0600-\u06FF]{3,}|[A-Za-z]+iyya(?:h|t)\b|\b\w+at\b', text)
    tokens = re.findall(r'[\u0600-\u06FF]+|\w+', text.lower())
    
    stopwords = {
        "the", "and", "what", "how", "when", "islam", "islamic", "ruling",
        "ال", "و", "في", "من", "هل", "ما", "عن", "اذا"
    }
    
    keywords = [
        token.strip(".,?!") 
        for token in tokens + islamic_terms
        if token not in stopwords and len(token) > 2
    ]
    
    return list(set(keywords))

# Document creation with enhanced metadata
def create_documents(qa_list: List[Dict[str, Any]]) -> List[Document]:
    documents = []
    for item in qa_list:
        text = f"Q: {item['question']}\nA: {item['answer']}"
        
        metadata = {
            "keywords": extract_keywords(item['question'] + " " + item['answer']),
            "source": item.get("source", "Unknown"),
            "category": item.get("category", "Uncategorized"),
            "is_general": "general:" in item['question'].lower()
        }
        
        documents.append(Document(text=text, metadata=metadata))
    return documents

# Initialize models
custom_embed_model = CustomEmbedding()

def get_llm():
    try:
        import requests
        requests.get("http://localhost:11434", timeout=5)
        return Ollama(
            model="gemma:2b",
            temperature=0.1,
            request_timeout=60.0,
            system_prompt="""You are an Islamic jurisprudential assistant. Rules:
1. Use only provided context
2. If unsure, say: "Please consult: alljnahalfkheah@yahoo.com"
3. Always cite sources
4. Preserve original Arabic terms"""
        )
    except Exception as e:
        print(f"Ollama connection error: {e}")
        exit(1)

llm = get_llm()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=custom_embed_model)

# Create index
documents = create_documents(qa_list)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Enhanced query processing
def process_query(query: str, index) -> str:
    """Universal query processing for all topics"""
    # First try keyword matching
    query_keywords = extract_keywords(query)
    keyword_nodes = []
    
    for node in index.docstore.docs.values():
        node_keywords = node.metadata.get("keywords", [])
        overlap = set(query_keywords) & set(node_keywords)
        if overlap:
            keyword_nodes.append((len(overlap), node))
    
    if keyword_nodes:
        keyword_nodes.sort(reverse=True)
        context_nodes = [node for (score, node) in keyword_nodes[:3] if score > 0]
        if context_nodes:
            return "\n\n".join([node.text for node in context_nodes])
    
    # Fallback to semantic search
    retriever = index.as_retriever(similarity_top_k=1)
    return retriever.retrieve(query)[0].node.text

# Response generation
def generate_response(query: str, context: str) -> str:
    prompt = f"""Islamic Jurisprudence Question Answering:
    
Context:
{context}

Question:
{query}

Rules:
1. Answer using ONLY the context
2. If answer isn't in context, respond: "Please consult: alljnahalfkheah@yahoo.com"
3. Include source when available

Answer:"""
    
    response = llm.complete(prompt)
    return response.text.strip()

# Verification
def verify_answer(answer: str) -> Dict[str, str]:
    answer_lower = answer.lower()
    if "consult" in answer_lower or "not in" in answer_lower:
        return {
            "answer": "Please consult the Jurisprudential Committee: alljnahalfkheah@yahoo.com",
            "source": "Unknown",
            "confidence": "Low"
        }
    
    # Check against all QA pairs
    for item in qa_list:
        if item["answer"].lower() in answer_lower:
            return {
                "answer": item["answer"],
                "source": item["source"],
                "confidence": "High"
            }
    
    return {
        "answer": answer,
        "source": "Various Sources",
        "confidence": "Medium"
    }

# Main interaction loop
print("Islamic Jurisprudence Assistant - Type 'exit' to quit")

while True:
    try:
        query = input("\nYour question: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        # Handle greetings
        if any(greet in query.lower() for greet in GREETINGS):
            print("As-salamu alaykum. How may I assist with Islamic jurisprudence?")
            continue

        # Process query
        context = process_query(query, index)
        answer = generate_response(query, context)
        verification = verify_answer(answer)
        
        # Format response
        response = f"\n{verification['answer']}"
        if verification['source'] != "Unknown":
            response += f"\n\nSource: {verification['source']}"
        print(response)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\nError: {e}\nPlease try again or rephrase your question")