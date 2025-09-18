import streamlit as st

from agentic_ai.agent import Agent
from agentic_ai.retriever import Retriever

st.set_page_config(page_title="agentic_ai - RAG Demo", layout="wide")
st.title("agentic_ai — Local RAG Demo")

with st.sidebar:
    st.header("Index")
    folder = st.text_input("Folder to index", "data/raw")
    if st.button("Build / Update Index"):
        with st.spinner("Indexing..."):
            Retriever().index_folder(folder)
        st.success("Index updated")

st.header("Ask")
q = st.text_input("Your question", "What is agentic RAG?")
k = st.slider("Top-K passages", 1, 10, 5)
if st.button("Ask"):
    with st.spinner("Thinking..."):
        res = Agent().ask(q, k=k)
    st.subheader("Answer")
    st.write(res["answer"])
    st.subheader("Sources")
    for i, s in enumerate(res["sources"], 1):
        st.markdown(f"**[{i}]** `{s['meta'].get('path','?')}` — score={s['score']:.3f}")
        with st.expander(f"Preview {i}"):
            st.write(s["text"])
