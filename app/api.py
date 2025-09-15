import os
import json
import requests
import streamlit as st
from langchain.memory import ConversationBufferMemory
from app.retriever import HybridRetriever
from app.config import OLLAMA_MODEL  # optional

# Initialize retriever and memory
retriever = HybridRetriever()
memory = ConversationBufferMemory(memory_key="chat_history")

# DeepInfra API setup
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    st.error("❌ API token not found. Set API_TOKEN in your environment.")
    st.stop()

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}
url = "https://api.deepinfra.com/v1/openai/chat/completions"

# Streamlit UI
st.title("OKR Key Results Generator")
objective = st.text_input("Enter your Objective")
top_k = st.number_input("Number of context chunks", min_value=1, max_value=50, value=10)

if st.button("Generate Key Results") and objective:
    # Retrieve relevant chunks
    chunks = retriever.retrieve(objective, top_k)
    context = "\n\n".join(c["text"] for c in chunks)

    # Load previous conversation
    past_memory = memory.load_memory_variables({}).get("chat_history", "")

    # Construct prompt
    prompt = (
        f"You are an AI assistant generating Key Results (KRs) for OKRs.\n\n"
        f"Objective: {objective}\n\n"
        f"Relevant context (optional inspiration):\n{context}\n\n"
        f"Rules:\n"
        f"- Suggest exactly 3 Key Results for the objective.\n"
        f"- Each KR must have a 'metric_type' field (achieved, milestone, numeric, percentage, currency).\n"
        f"- For 'milestone' type, include a 'milestones' list with sub-milestones, each with title and weight.\n"
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
        st.error(f"DeepInfra API call failed: {response.text}")
    else:
        answer_raw = response.json()["choices"][0]["message"]["content"]
        # Parse JSON safely
        try:
            answer_json = json.loads(answer_raw)
        except json.JSONDecodeError:
            answer_raw_clean = answer_raw.strip("` \n")
            if answer_raw_clean.startswith("{") and answer_raw_clean.endswith("}"):
                answer_json = json.loads(answer_raw_clean)
            else:
                st.error(f"Invalid JSON from model:\n{answer_raw}")
                answer_json = None

        if answer_json:
            # Save conversation in memory
            memory.save_context({"input": objective}, {"output": json.dumps(answer_json)})

            # Display results
            st.subheader("Generated Key Results")
            st.json(answer_json)

            st.subheader("Relevant Context Chunks")
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}:** {c['text']}")

            st.subheader("Conversation Memory")
            st.json(memory.load_memory_variables({}))
