import streamlit as st
import streamlit.components.v1 as components

def render_copy_button(text: str, key: str):
    """Render a small copy-to-clipboard button for a chat response."""
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$").replace("</", "<\\/")
    components.html(
        f"""
        <style>body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; }}</style>
        <button id="copy-{key}" style="
            all: unset;
            cursor: pointer;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #8ea8b8;
            border: 1px solid rgba(150,190,210,0.2);
            border-radius: 6px;
            padding: 4px 10px;
            background: rgba(14,28,40,0.8);
            transition: all 160ms ease;
        " onmouseover="this.style.borderColor='rgba(111,209,199,0.4)'; this.style.color='#c8f2eb';"
           onmouseout="this.style.borderColor='rgba(150,190,210,0.2)'; this.style.color='#8ea8b8';">
            Copy
        </button>
        <script>
            document.getElementById('copy-{key}').addEventListener('click', function() {{
                navigator.clipboard.writeText(`{escaped}`).then(() => {{
                    const btn = this;
                    btn.innerText = 'Copied';
                    btn.style.borderColor = '#38b7a6';
                    btn.style.color = '#6fd1c7';
                    setTimeout(() => {{
                        btn.innerText = 'Copy';
                        btn.style.borderColor = 'rgba(150,190,210,0.2)';
                        btn.style.color = '#8ea8b8';
                    }}, 1800);
                }});
            }});
        </script>
        """,
        height=32,
    )
