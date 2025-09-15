import os
import json
import requests
import streamlit as st
from langchain.memory import ConversationBufferMemory
from app.retriever import HybridRetriever

# ----------------------
# Configuration
# ----------------------
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    st.error("❌ API token not found. Set API_TOKEN in your environment.")
    st.stop()

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}
URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# Directory to save chat sessions
CHAT_SAVE_DIR = "chat_sessions"
os.makedirs(CHAT_SAVE_DIR, exist_ok=True)

# ----------------------
# Initialize Retriever + Memory
# ----------------------
retriever = HybridRetriever()
memory = ConversationBufferMemory(memory_key="chat_history")

# ----------------------
# Helper to parse JSON if returned
# ----------------------
def parse_model_json(answer_raw):
    try:
        return json.loads(answer_raw)
    except json.JSONDecodeError:
        start = answer_raw.find("{")
        end = answer_raw.rfind("}")
        if start != -1 and end != -1:
            json_str = answer_raw[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
        return None

# ----------------------
# Generate chatbot reply
# ----------------------
def generate_reply(query):
    chunks = retriever.retrieve(query)
    context_text = "\n\n".join(c["text"] for c in chunks) if chunks else ""
    past_memory = memory.load_memory_variables({}).get("chat_history", "")

    prompt = (
        "You are a helpful AI assistant.\n"
        "Use the following context to answer the user. "
        "If the context is insufficient, answer based on your general knowledge.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Conversation history:\n{past_memory}\n\n"
        f"User question: {query}\nAssistant:"
    )

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }

    response = requests.post(URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        return f"❌ API Error: {response.text}"

    answer_raw = response.json()["choices"][0]["message"]["content"]
    parsed_answer = parse_model_json(answer_raw)
    return parsed_answer if parsed_answer else answer_raw

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("📩 AI Chatbot")

# ----------------------
# Session state
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_name" not in st.session_state:
    st.session_state.session_name = None
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

# ----------------------
# Sidebar: New chat + saved chats + delete
# ----------------------
st.sidebar.header("💾 Chat Sessions")

if st.sidebar.button("➕ New Chat"):
    st.session_state.messages = []
    st.session_state.session_name = None
    memory.clear()

# Saved chats
saved_chats = sorted([f for f in os.listdir(CHAT_SAVE_DIR) if f.endswith(".json")])

for chat_file in saved_chats:
    chat_name = chat_file.replace(".json", "").replace("_", " ")
    col1, col2 = st.sidebar.columns([4, 1], gap="small")

    with col1:
        if st.button(chat_name, key=f"load_{chat_name}"):
            with open(os.path.join(CHAT_SAVE_DIR, chat_file), "r") as f:
                st.session_state.messages = json.load(f)
            st.session_state.session_name = chat_name

    with col2:
        if st.button("🗑️", key=f"del_{chat_name}"):
            st.session_state.confirm_delete = chat_name

# Confirm delete popup
if st.session_state.confirm_delete:
    st.sidebar.warning(f"Delete chat '{st.session_state.confirm_delete}'?")
    colA, colB = st.sidebar.columns(2)
    with colA:
        if st.button("✅ Yes, Delete"):
            filename = st.session_state.confirm_delete.replace(" ", "_") + ".json"
            path = os.path.join(CHAT_SAVE_DIR, filename)
            if os.path.exists(path):
                os.remove(path)
            if st.session_state.session_name == st.session_state.confirm_delete:
                st.session_state.messages = []
                st.session_state.session_name = None
            st.session_state.confirm_delete = None
            st.rerun()  # ✅ FIXED here
    with colB:
        if st.button("❌ Cancel"):
            st.session_state.confirm_delete = None
            st.rerun()  # ✅ FIXED here

# ----------------------
# Display previous messages
# ----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------
# Chat input always at bottom
# ----------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # If new chat: assign name from the first user message using spaces
    if st.session_state.session_name is None:
        short_name = " ".join(user_input.strip().split()[:5])
        st.session_state.session_name = short_name

    # Show user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI reply
    with st.spinner("AI is typing..."):
        reply = generate_reply(user_input)
        memory.save_context({"input": user_input}, {"output": reply})
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

    # Save or update chat file
    filename = st.session_state.session_name.replace(" ", "_") + ".json"
    save_path = os.path.join(CHAT_SAVE_DIR, filename)
    with open(save_path, "w") as f:
        json.dump(st.session_state.messages, f, indent=2)
