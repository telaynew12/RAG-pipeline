from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class LocalLLM:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        local_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        self.model = HuggingFacePipeline(pipeline=local_pipeline)

    def create_qa_chain(self, retriever):
        return RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=retriever,
            chain_type="stuff"
        )
