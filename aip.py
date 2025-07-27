import json
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import BaseEmbedding  # Updated import path

# Load Q&A from file
with open("qa_data.json", "r", encoding="utf-8") as f:
    qa_list = json.load(f)

documents = []
for item in qa_list:
    q = item["question"]
    a = item["answer"]
    full_text = f"Q: {q}\nA: {a}"
    documents.append(Document(text=full_text))

# Custom Embedding Class (now compatible with latest llama-index)
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

# Initialize components
custom_embed_model = CustomEmbedding()
llm = Ollama(model="mistral")  # Ensure Ollama is running locally!

# Configure global Settings (replaces ServiceContext)
Settings.llm = llm
Settings.embed_model = custom_embed_model
Settings.node_parser = SimpleNodeParser()

# Build index
nodes = Settings.node_parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)  # No need for service_context

# Query engine
query_engine = index.as_query_engine()

print("Ask your questions! Type 'exit' to quit.")
while True:
    query = input("Your question: ")
    if query.lower() == "exit":
        break
    response = query_engine.query(query)
    print(f"\nAnswer:\n{response}\n")