import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient

from retrieval import run_retrieval
from ui.style import load_css
from ui.components import render_top_strip, render_welcome_screen
from ui.sidebar import render_sidebar

st.set_page_config(
    page_title="GovAssist | Government RAG Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
load_css()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

def queue_prompt(prompt_text: str) -> None:
    st.session_state.queued_prompt = prompt_text

def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.queued_prompt = None

load_dotenv()

@st.cache_resource(show_spinner=False)
def get_clients() -> tuple[genai.Client, QdrantClient]:
    """Create singleton clients to avoid local Qdrant lock collisions on reruns."""
    gemini = genai.Client()
    qdrant = QdrantClient(path="local_qdrant_db")
    return gemini, qdrant

try:
    gemini_client, qdrant_client = get_clients()
except Exception as e:
    err_text = str(e)
    if "already accessed by another instance" in err_text:
        st.error(
            "Failed to initialize clients: local_qdrant_db is locked by another running process. "
            "Stop any other Python/Streamlit process using this folder, then restart this app."
        )
    else:
        st.error(f"Failed to initialize clients: {e}")
    st.stop()

# Render Sidebar
render_sidebar(clear_chat, queue_prompt)

# Main App Shell
st.markdown('<div class="shell">', unsafe_allow_html=True)

gen_model = os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it")
embed_model = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")

# Render Top Strip
render_top_strip(gen_model, embed_model)

# Render Welcome Screen if no messages
if not st.session_state.messages:
    render_welcome_screen()

# Render Chat History
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🛰️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle Input
typed_prompt = st.chat_input("Ask anything from your government docs knowledge base...")
prompt = st.session_state.queued_prompt if st.session_state.queued_prompt else typed_prompt
st.session_state.queued_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🛰️"):
        with st.spinner("Running hybrid retrieval and drafting grounded answer..."):
            try:
                answer_stream = run_retrieval(
                    gemini_client=gemini_client,
                    qdrant_client=qdrant_client,
                    query=prompt,
                    collection_name="gov_docs",
                    chat_history=st.session_state.chat_history,
                )

                if answer_stream is None:
                    response_text = (
                        "Sorry, I could not find an exact answer in the indexed government documents."
                    )
                    st.markdown(response_text)
                else:
                    response_text = st.write_stream(
                        chunk.text for chunk in answer_stream if chunk.text
                    )

                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.chat_history.append({"role": "user", "text": prompt})
                st.session_state.chat_history.append({"role": "model", "text": response_text})
            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")

st.markdown(
    '<div class="footer-note">GovAssist may generate mistakes. Verify with official published circulars.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
