import os
import base64
from pathlib import Path
import streamlit as st

def get_logo_html() -> str:
    """Return an img tag for brand logo if found, otherwise fallback initials."""
    explicit_logo_path = os.getenv("GOVASSIST_LOGO_PATH", "").strip()
    candidate_paths = []

    if explicit_logo_path:
        candidate_paths.append(Path(explicit_logo_path))

    candidate_paths.extend(
        [
            Path("assets/logo.png"),
            Path("assets/logo.jpg"),
            Path("assets/logo.jpeg"),
            Path("assets/logo.webp"),
            Path("logo.png"),
        ]
    )

    for logo_path in candidate_paths:
        if logo_path.exists() and logo_path.is_file():
            ext = logo_path.suffix.lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(ext)
            if not mime_type:
                continue

            encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            return (
                f'<img src="data:{mime_type};base64,{encoded}" '
                f'alt="GovAssist Logo" class="brand-mark-logo" />'
            )

    return '<span class="brand-mark-fallback">GA</span>'

def render_top_strip(gen_model: str, embed_model: str):
    """Render the top brand and model badge strip."""
    brand_logo_html = get_logo_html()
    st.markdown(
        f"""
<div class="top-strip">
    <div class="brand">
        <div class="brand-mark">{brand_logo_html}</div>
        <div>
            <div class="brand-name">GovAssist Frontier</div>
            <div class="brand-note">PRODUCTION RAG CHAT SURFACE</div>
        </div>
    </div>
    <div class="badge-row">
        <div class="badge">GEN · {gen_model}</div>
        <div class="badge">EMBED · {embed_model}</div>
        <div class="badge">VECTOR STORE · QDRANT LOCAL</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

def render_welcome_screen():
    """Render the initial empty state welcome screen with feature cards."""
    st.markdown(
        """
<div class="welcome">
    <h2 class="welcome-title">Ask policy, rules, and notifications with <span class="welcome-title-accent">citation-ready grounding</span>.</h2>
    <p class="welcome-copy">Responses are generated strictly from your indexed government documents, then structured for clarity and decision workflows.</p>
    <div class="welcome-grid">
        <div class="welcome-card">
            <div class="welcome-card-title">Grounded Answers</div>
            <div class="welcome-card-copy">No hallucinated policy text. Retrieval-first reasoning from your local corpus.</div>
        </div>
        <div class="welcome-card">
            <div class="welcome-card-title">Comparative Analysis</div>
            <div class="welcome-card-copy">Contrast clauses, eligibility rules, and revision notices in one response.</div>
        </div>
        <div class="welcome-card">
            <div class="welcome-card-title">Source Confidence</div>
            <div class="welcome-card-copy">Every answer is designed to include explainability and source-backed context.</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.info("Use the quick actions in the sidebar to launch common government-doc queries.")
