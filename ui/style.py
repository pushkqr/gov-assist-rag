from pathlib import Path
import streamlit as st

def load_css():
    """Reads style.css from disk and injects it into Streamlit."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)
    else:
        st.warning("Could not find style.css")
