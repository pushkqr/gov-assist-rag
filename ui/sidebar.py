import streamlit as st

def render_sidebar(clear_chat_callback, queue_prompt_callback):
    """Render the sidebar with quick actions and stats."""
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-hero">
    <div class="sidebar-title">GovAssist Control Room</div>
    <div class="sidebar-subtle">Grounded responses from indexed government PDFs, tuned for citation-backed policy workflows.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.divider()

        st.metric("Messages", value=len(st.session_state.messages))
        st.metric("Turns in memory", value=len(st.session_state.chat_history) // 2)

        if st.button("New chat", use_container_width=True):
            clear_chat_callback()
            st.rerun()

        st.caption("Tip: ask focused questions with year or department names for higher precision.")
        st.divider()
        st.markdown("**Quick actions**")
        if st.button("Summarize latest notification", use_container_width=True, key="sidebar_summarize"):
            queue_prompt_callback("Summarize the most recent notification in simple bullet points.")
            st.rerun()
        if st.button("Eligibility criteria checklist", use_container_width=True, key="sidebar_eligibility"):
            queue_prompt_callback("Create an eligibility criteria checklist from the relevant government document.")
            st.rerun()
        if st.button("Compare two policy clauses", use_container_width=True, key="sidebar_compare"):
            queue_prompt_callback("Compare the two most relevant clauses related to this policy and explain the difference.")
            st.rerun()
