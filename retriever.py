from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(self.embedding_model.get_sentence_embedding_dimension())
        self.document_store = {}

    def load_documents(self):
        
        # load from text files in the /data
        documents = []
        with open('data/documents.txt', 'r') as f:
            documents = f.readlines()
        documents = [doc.strip() for doc in documents]
        
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        self.index.add(embeddings)
        for i, doc in enumerate(documents):
            self.document_store[i] = doc
        return self.document_store

    def retrieve(self, query):
        user_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(user_embedding, k=2)
        retrieved_docs = "\n".join([
            self.document_store[idx] for idx in I[0] if idx != -1 and idx in self.document_store
        ])
        return retrieved_docs
