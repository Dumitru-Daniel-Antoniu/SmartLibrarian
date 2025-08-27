import streamlit as st

from src.config import settings
from src.llm.chat_orchestrator import answer_user_query


"""
Streamlit web app for SmartLibrarian: an interactive book recommendation chatbot.
Displays chat interface, handles user queries, and shows semantic search results from the library.
"""


st.set_page_config(page_title="SmartLibrarian", page_icon="📚")

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.write(f"**Chat model:** `{settings.CHAT_MODEL}`")
    st.write(f"**Embed model:** `{settings.EMBED_MODEL}`")
    st.write(f"**Top-K (RAG):** `{settings.TOP_K}`")
    if st.button("Clear chat"):
        st.session_state.clear()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Tell me what kind of book you're looking for-"
                "e.g., *friendship and magic*, *quiet sci-fi* or *adventure stories*"
            )
        }
    ]
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []

st.title("📚 SmartLibrarian")
st.caption("Type what you enjoy (themes, genres, etc.). I'll search the library and recommend one book.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_text = st.chat_input("What kind of book should I find for you?")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    result = answer_user_query(user_text)
    assistant_text = result.get("final_text", "Sorry, I couldn't generate a response.")
    st.session_state.last_hits = result.get("hits", [])

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

        hits = st.session_state.last_hits or []
        if hits:
            with st.expander("🔎 RAG debug: top matches"):
                for i, h in enumerate(hits, start=1):
                    title = h.get("title", "(untitled)")
                    distances = h.get("distance", None)
                    first_line = (h.get("summary", "").splitlines() or [""])[0]
                    if distances is not None:
                        st.write(f"**{i}) {title}** \n_distance: {distances:.4f}_")
                    else:
                        st.write(f"**{i}) {title}**")
                    st.caption(first_line)

st.markdown("---")
st.caption(
    "Tip: try queries like *I want a book about liberty and social control.*, "
    "*What do you recommend me if I love fantastic stories?* or "
    "*What is 1984?*"
)
