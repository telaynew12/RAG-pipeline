from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

class DocumentChunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
