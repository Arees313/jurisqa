import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

# Load your data
with open('qa_data.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# Step 1: Improved text generation with metadata weighting
indexed_texts = []
for item in qa_data:
    keywords = " ".join([f"{kw} " * 3 for kw in item.get("keywords", [])])  # 3x weight
    tags = " ".join([f"{tag} " * 2 for tag in item.get("tags", [])])  # 2x weight
    category = item.get("category", "") + " " * 2  # 2x weight
    question = item.get("question", "")
    answer = item.get("answer", "")
    text = f"{keywords}{tags}{category}Question: {question} Answer: {answer}"
    indexed_texts.append(text.lower())  # Simple normalization

# Load model
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Generate new embeddings
print("Generating new embeddings...")
embeddings = model.encode(indexed_texts, show_progress_bar=True)

# Save new index
print("Saving new index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, "qa.index")
np.save("qa_embeddings.npy", embeddings)

print("✅ Index rebuilt successfully!")