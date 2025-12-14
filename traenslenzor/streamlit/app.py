import asyncio

import streamlit as st

from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.logger import setup_logger
from traenslenzor.supervisor.supervisor import run

setup_logger()
st.set_page_config(layout="wide")

if "history" not in st.session_state:
    st.session_state.history = [
        {
            "role": "assistant",
            "content": "Document Assistant Ready! I can help you with document operations. Please provide a document:",
        }
    ]

if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = None


def chat_stream(text: str):
    for ch in text:
        yield ch
        # time.sleep(0.02)


def fetch_image(session_id: str):
    sess = asyncio.run(SessionClient.get(session_id))
    file_ref = sess.extractedDocument
    if not file_ref:
        return None
    return asyncio.run(FileClient.get_image(file_ref.id))


# Read input FIRST (chat_input may be full width, so keep it outside columns)
prompt = st.chat_input("Say something")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})

    msg, session_id = asyncio.run(run(prompt))
    st.session_state.last_session_id = str(session_id)

    st.session_state.history.append({"role": "assistant", "content": getattr(msg, "content")})
    st.rerun()

chat_col, img_col = st.columns([1, 1], gap="large")

with chat_col:
    st.title("TrÄenslÄnzÖr 0815 Döküment Äsißtänt")
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.write_stream(chat_stream(message["content"]))
            else:
                st.write(message["content"])

with img_col:
    if st.session_state.last_session_id:
        img = fetch_image(st.session_state.last_session_id)
        if img is not None:
            st.image(img, use_container_width=True)
