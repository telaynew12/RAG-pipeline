import os
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List

class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None

    def create_store(self, documents: List[Document], persist_dir: str = "./chroma_db") -> Chroma:
        # Check if vectorstore already exists
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
            print(f"✅ Loaded existing vectorstore from {persist_dir}")
        else:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            self.vectorstore.persist()
            print(f"✅ Vectorstore created and persisted to {persist_dir}")
        return self.vectorstore
