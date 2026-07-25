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

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", value=len(st.session_state.messages))
        with col2:
            st.metric("Memory turns", value=len(st.session_state.chat_history) // 2)

        if st.button("New chat", use_container_width=True, type="primary"):
            clear_chat_callback()
            st.rerun()

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

        st.divider()
        st.caption("Tip: include year or department names in your query for higher precision retrieval.")
