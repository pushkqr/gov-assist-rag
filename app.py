import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient

from core.utils import get_genai_client
from retrieval import run_retrieval
from ui.style import load_css
from ui.components import render_top_strip, render_welcome_screen
from ui.sidebar import render_sidebar
from ui.copy_button import render_copy_button

load_dotenv()

st.set_page_config(
    page_title="GovAssist | Government RAG Assistant",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

SESSION_FILE = Path("temp/chat_session.json")

def _load_session():
    """Load persisted chat session from disk if available."""
    if SESSION_FILE.exists():
        try:
            data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            return data.get("messages", []), data.get("chat_history", [])
        except (json.JSONDecodeError, KeyError):
            return [], []
    return [], []

def _save_session():
    """Persist current chat session to disk."""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(
        json.dumps({
            "messages": st.session_state.messages,
            "chat_history": st.session_state.chat_history,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

if "messages" not in st.session_state:
    saved_msgs, saved_history = _load_session()
    st.session_state.messages = saved_msgs
    st.session_state.chat_history = saved_history
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
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()

@st.cache_resource(show_spinner=False)
def get_clients() -> tuple[genai.Client, QdrantClient]:
    """Create singleton clients to avoid local Qdrant lock collisions on reruns."""
    gemini = get_genai_client()
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

# ── Error Formatting ──
def _friendly_error(exc: Exception) -> str:
    """Convert raw exceptions into user-friendly messages."""
    msg = str(exc).lower()
    if "getaddrinfo" in msg or "name resolution" in msg or "connection" in msg:
        return "Unable to reach the AI service. Please check your internet connection and try again."
    if "429" in msg or "quota" in msg or "exhausted" in msg:
        return "The system is currently rate-limited. Please wait a moment and try again."
    if "api key" in msg or "permission" in msg or "401" in msg or "403" in msg:
        return "Authentication error. Please verify your API key configuration."
    return f"An unexpected error occurred: {exc}"

# ── Layout ──
render_sidebar(clear_chat, queue_prompt)

st.markdown('<div class="shell">', unsafe_allow_html=True)

gen_model = os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it")
embed_model = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")

render_top_strip(gen_model, embed_model)


# Welcome Screen
if not st.session_state.messages:
    render_welcome_screen()

# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    avatar = "assets/user.jpg" if msg["role"] == "user" else "assets/logo.png"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_copy_button(msg["content"], key=f"history-{idx}")

# Handle Input
typed_prompt = st.chat_input("Ask anything from your government docs knowledge base...")
prompt = st.session_state.queued_prompt if st.session_state.queued_prompt else typed_prompt
st.session_state.queued_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="assets/user.jpg"):
        st.markdown(prompt)

    try:
        # Step 1: Embedding & Search
        with st.spinner("Assistant is processing the query..."):
            retrieval_result = run_retrieval(
                gemini_client=gemini_client,
                qdrant_client=qdrant_client,
                query=prompt,
                collection_name="gov_docs",
                chat_history=st.session_state.chat_history,
            )

        with st.chat_message("assistant", avatar="assets/logo.png"):
            if isinstance(retrieval_result, dict):
                if retrieval_result.get("status") in ("empty", "error"):
                    response_text = retrieval_result.get("response_text", "An error occurred.")
                    st.markdown(response_text)
                else:
                    answer_stream = retrieval_result.get("answer_stream")
                    response_text = st.write_stream(answer_stream)
            else:
                if retrieval_result is None:
                    response_text = (
                        "Sorry, I could not find an exact answer in the indexed government documents."
                    )
                    st.markdown(response_text)
                else:
                    response_text = st.write_stream(
                        (chunk.text if hasattr(chunk, "text") else chunk) for chunk in retrieval_result if chunk
                    )

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.chat_history.append({"role": "user", "text": prompt})
            st.session_state.chat_history.append({"role": "model", "text": response_text})

            render_copy_button(response_text, key=f"live-{len(st.session_state.messages)}")

            _save_session()

    except Exception as e:
        st.error(_friendly_error(e))

st.markdown(
    '<div class="footer-note">GovAssist may generate mistakes. Verify with official published circulars.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
