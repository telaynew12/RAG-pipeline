from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from typing import List

class WebContentLoader:
    def __init__(self, urls: List[str]):
        self.urls = urls

    def load_content(self) -> List[Document]:
        loader = WebBaseLoader(self.urls)
        try:
            documents = loader.load()
            print(f"✅ Successfully loaded {len(documents)} documents from {len(self.urls)} URLs")
            return documents
        except Exception as e:
            print(f"❌ Error loading content: {e}")
            return []
