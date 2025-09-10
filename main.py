from pipeline.loader import WebContentLoader
from pipeline.chunker import DocumentChunker
from pipeline.embeddings import LocalEmbeddings
from pipeline.vectorstore import VectorStore
from pipeline.llm import LocalLLM

def build_pipeline(urls):
    # Load documents
    loader = WebContentLoader(urls)
    documents = loader.load_content()

    # Chunk documents
    chunker = DocumentChunker()
    chunks = chunker.create_chunks(documents)

    # Create embeddings and vectorstore
    embeddings = LocalEmbeddings().get_embeddings()
    vectorstore = VectorStore(embeddings).create_store(chunks)

    # Create retriever and QA chain
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    llm = LocalLLM()
    qa_chain = llm.create_qa_chain(retriever)
    return qa_chain

if __name__ == "__main__":
    urls = [
        "https://www.geeksforgeeks.org/nlp/stock-price-prediction-project-using-tensorflow/",
        "https://www.geeksforgeeks.org/deep-learning/training-of-recurrent-neural-networks-rnn-in-tensorflow/"
    ]

    qa_chain = build_pipeline(urls)

    print("\n✅ Chatbot ready! Type your question (type 'exit' to quit):\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        # Debug: show retrieved chunks
        docs = qa_chain.retriever.get_relevant_documents(query)
        print("\nRetrieved Chunks:")
        for i, doc in enumerate(docs):
            print(f"{i+1}: {doc.page_content[:150]}...\n")

        # Generate answer
        response = qa_chain.invoke({"query": query})
        print(f"Bot: {response['result']}\n")
