# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # from langchain.memory import ConversationBufferMemory
# # from app.retriever import HybridRetriever
# # import subprocess
# # from app.config import OLLAMA_MODEL

# # app = FastAPI()
# # retriever = HybridRetriever()
# # memory = ConversationBufferMemory(memory_key="chat_history")

# # class QueryRequest(BaseModel):
# #     query: str
# #     top_k: int = 10

# # API_TOKEN = os.getenv("API_TOKEN")
# # if not API_TOKEN:
# #     st.error("❌ API token not found.")
# #     st.stop()

# # headers = {
# #     "Content-Type": "application/json",
# #     "Authorization": f"Bearer {API_TOKEN}"
# # }
# # url = "https://api.deepinfra.com/v1/openai/chat/completions"

# # @app.post("/chat")
# # def chat(req: QueryRequest):
# #     chunks = retriever.retrieve(req.query, req.top_k)
# #     context = "\n\n".join(c["text"] for c in chunks)
# #     memory.save_context({"input": req.query}, {"output": context})

# #     if OLLAMA_MODEL:
# #         prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {req.query}"
# #         res = subprocess.run(
# #             ["ollama", "run", OLLAMA_MODEL],
# #             input=prompt.encode(),
# #             capture_output=True
# #         )
# #         return {"answer": res.stdout.decode(), "chunks": chunks, "memory": memory.load_memory_variables({})}
# #     return {"context": context, "chunks": chunks, "memory": memory.load_memory_variables({})}




# import os
# import requests
# from fastapi import FastAPI
# from pydantic import BaseModel
# from langchain.memory import ConversationBufferMemory
# from app.retriever import HybridRetriever
# from app.config import OLLAMA_MODEL  # keep if you still want to support Ollama

# app = FastAPI()
# retriever = HybridRetriever()
# memory = ConversationBufferMemory(memory_key="chat_history")

# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 10

# API_TOKEN = os.getenv("API_TOKEN")
# if not API_TOKEN:
#     raise RuntimeError("❌ API token not found. Set API_TOKEN in your environment.")

# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {API_TOKEN}"
# }
# url = "https://api.deepinfra.com/v1/openai/chat/completions"

# @app.post("/chat")
# def chat(req: QueryRequest):
#     # Retrieve chunks for current query
#     chunks = retriever.retrieve(req.query, req.top_k)
#     context = "\n\n".join(c["text"] for c in chunks)

#     # Load past conversation from memory
#     past_memory = memory.load_memory_variables({}).get("chat_history", "")

#     # Build prompt with past chat + retrieved context
#     prompt = (
#         # f"You are an AI assistant helping the user based on previous conversation and context.\n\n"
#         # f"Previous conversation:\n{past_memory}\n\n"
#         # f"Relevant context:\n{context}\n\n"
#         # f"User's new question: {req.query}\n"
#         # f"Answer accordingly."
        


#         f"You are an AI assistant helping the user based on context.\n\n"
#         f"Relevant context:\n{context}\n\n"
#         f"Suggest 3 measurable key results for the following objective:{req.query}, based on how okr is structured\n\n"
#         f"Metrics supported: Achieved/Not, Milestone, Currency, Numeric, Percentage.\n\n"
#         f"Return results in JSON with keys: title, metric_type,  initial_value, target_value, weight. based on the type of the key result\n\n"
#     )

#     # Call DeepInfra API
#     payload = {
#         "model": "mistralai/Mistral-7B-Instruct-v0.2",  # change to your DeepInfra model
#         "messages": [
#             {"role": "system", "content": "You are a helpful and knowledgeable copilot."},
#             {"role": "user", "content": prompt}
#         ],
#         "max_tokens": 2048,
#         "temperature": 0.7
#     }
#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code != 200:
#         return {"error": f"DeepInfra API call failed: {response.text}"}

#     result = response.json()
#     answer = result["choices"][0]["message"]["content"]

#     # Save the actual conversation turn to memory
#     memory.save_context({"input": req.query}, {"output": answer})

#     return {
#         "answer": answer,
#         "chunks": chunks,
#         "memory": memory.load_memory_variables({})
#     }
import os
import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from app.retriever import HybridRetriever
from app.config import OLLAMA_MODEL  # optional if you still want to support Ollama

app = FastAPI()
retriever = HybridRetriever()
memory = ConversationBufferMemory(memory_key="chat_history")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("❌ API token not found. Set API_TOKEN in your environment.")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}
url = "https://api.deepinfra.com/v1/openai/chat/completions"

@app.post("/chat")
def chat(req: QueryRequest):
    # Retrieve relevant chunks
    chunks = retriever.retrieve(req.query, req.top_k)
    context = "\n\n".join(c["text"] for c in chunks)

    # Load previous conversation
    past_memory = memory.load_memory_variables({}).get("chat_history", "")

    # Construct strict JSON-only prompt for Key Results
    prompt = (
    f"You are an AI assistant generating Key Results (KRs) for OKRs.\n\n"
    f"Objective: {req.query}\n\n"
    f"Relevant context (optional inspiration):\n{context}\n\n"
    f"Rules:\n"
    f"- Suggest exactly 3 Key Results for the objective.\n"
    f"- Each KR must have a 'metric_type' field (achieved, milestone, numeric, percentage, currency).\n"
    f"- For 'milestone' type, include a 'milestones' list with sub-milestones, each with title and weight. "
    f"Do NOT include initial_value or target_value for milestones.\n"
    f"- For 'achieved': do NOT include initial_value or target_value and do NOT include 'milestones' and do NOT include status.\n"
    f"- For 'numeric', 'percentage', 'currency': include initial_value and target_value.\n"
    f"- Parent KR weight should equal the sum of sub-milestones weights (if milestone).\n"
    f"- Return ONLY valid JSON like:\n"
    f'{{ "Key Results": [ {{...}}, {{...}}, {{...}} ] }}'
   )



    # Call DeepInfra API
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "system", "content": "You are a JSON-only OKR copilot."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": f"DeepInfra API call failed: {response.text}"}

    result = response.json()
    answer_raw = result["choices"][0]["message"]["content"]

    # Parse JSON safely
    try:
        answer_json = json.loads(answer_raw)
    except json.JSONDecodeError:
        answer_raw_clean = answer_raw.strip("` \n")
        if answer_raw_clean.startswith("{") and answer_raw_clean.endswith("}"):
            answer_json = json.loads(answer_raw_clean)
        else:
            return {"error": "Invalid JSON from model", "raw_answer": answer_raw}

    # Save conversation in memory
    memory.save_context({"input": req.query}, {"output": json.dumps(answer_json)})

    return {
        "answer": answer_json,  # structured dict for frontend
        "chunks": chunks,
        "memory": memory.load_memory_variables({})
    }


from fastapi import Query

@app.post("/regenerate")
def regenerate(query: str = Query(...), top_k: int = 10, ignore_memory: bool = False):
    # Retrieve chunks
    chunks = retriever.retrieve(query, top_k)
    context = "\n\n".join(c["text"] for c in chunks)

    # Load previous conversation if not ignoring memory
    past_memory = "" if ignore_memory else memory.load_memory_variables({}).get("chat_history", "")

    prompt = (
        f"You are an AI assistant generating Key Results (KRs) for OKRs.\n\n"
        f"Objective: {query}\n\n"
        f"Relevant context:\n{context}\n\n"
        f"Previous conversation (optional):\n{past_memory}\n\n"
        f"Rules:\n"
        f"- Suggest exactly 3 Key Results.\n"
        f"- For milestone type, include sub-milestones with title and weight.\n"
        f"- For achieved/not achieved, no initial/target, no milestones.\n"
        f"- For numeric/percentage/currency, include initial_value and target_value.\n"
        f"- Return ONLY valid JSON like {{'Key Results':[...]]}}"
    )

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "system", "content": "You are a JSON-only OKR copilot."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7  # higher temperature = more varied regenerations
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return {"error": f"DeepInfra API call failed: {response.text}"}

    answer_raw = response.json()["choices"][0]["message"]["content"]

    # Parse JSON safely
    try:
        answer_json = json.loads(answer_raw)
    except json.JSONDecodeError:
        answer_raw_clean = answer_raw.strip("` \n")
        if answer_raw_clean.startswith("{") and answer_raw_clean.endswith("}"):
            answer_json = json.loads(answer_raw_clean)
        else:
            return {"error": "Invalid JSON from model", "raw_answer": answer_raw}

    # Save regenerated response in memory
    memory.save_context({"input": query}, {"output": json.dumps(answer_json)})

    return {
        "answer": answer_json,
        "chunks": chunks,
        "memory": memory.load_memory_variables({})
    }
