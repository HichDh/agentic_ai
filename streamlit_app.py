import os
import torch
import streamlit as st
from src.agentic_ai.lc_pipeline import build_vectorstore, build_rag_chain
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from agentic_ai.config import settings
from langchain_core.messages import HumanMessage, AIMessage


@st.cache_resource(show_spinner=False)
def load_llm(model_id: str, device_pref: str):

    # resolve device (keep it simple: settings has "cpu" or "mps")
    device = device_pref if device_pref in {"mps", "cpu"} else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,     # safe & fast enough on MPS/CPU
        low_cpu_mem_usage=True,
    )
    if device == "mps" and torch.backends.mps.is_available():
        model.to("mps")
    model.eval()
    return tokenizer, model

st.set_page_config(page_title="agentic_ai - RAG Demo", layout="wide")
st.title("agentic_ai — Local RAG Demo")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "human"/"ai", "content": "..."}

index_path = str(settings.index_path)
print("settings.generator_model: ", settings.generator_model)

def index_exists():
    return os.path.exists(index_path) and os.path.isdir(index_path)

with st.sidebar:
    st.header("Index")
    folder = st.text_input("Folder to index", "data/raw")
    if st.button("Build / Update Index"):
        with st.spinner("Indexing..."):
            vs = build_vectorstore(folder)
            vs.save_local(index_path)
            st.session_state["vs"] = vs
        st.success("Index updated")

    st.header("Generation")
    default_do_sample = getattr(settings, "do_sample_default", False)
    default_temp = getattr(settings, "temperature_default", 0.7)
    default_max_new = getattr(settings, "max_new_tokens", 192)

    use_sampling = st.checkbox("Use sampling", value=default_do_sample, key="use_sampling")
    temperature = st.slider("Temperature", 0.1, 1.5, default_temp, 0.1, key="temperature")
    max_new = st.slider("Max new tokens", 64, 512, default_max_new, 32, key="max_new_tokens")

    # --- Guardrails ---
    st.header("Guardrails")
    min_score = st.slider("Min relevance", 0.0, 1.0, 0.30, 0.01, key="min_score")
    strict_mode = st.checkbox("Strict mode", value=True, key="strict_mode") # (does not use sources if below threshold)

    if st.button("Clear chat"):
        st.session_state.chat_history = []

if "vs" not in st.session_state:
    if index_exists():
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        print(f"Loading Index {index_path} ...")
        st.session_state["vs"] = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        st.session_state["vs"] = None

st.header("Ask")
q = st.text_input("Your question", "What is agentic RAG?")
k = st.slider("Top-K passages", 1, 10, 5)
if st.button("Ask"):
    with st.spinner("Thinking..."):
        vs = st.session_state.get("vs")
        if not vs:
            st.error("Please build the index first.")
        else:
            # Load generator ONCE (cached)
            tokenizer, model = load_llm(settings.generator_model, settings.device)

        # Build RAG chain with this LLM
        gen_kwargs = dict(
            max_new_tokens=int(st.session_state["max_new_tokens"]),
            do_sample=bool(st.session_state["use_sampling"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
            use_cache=True,
            temperature=(st.session_state["temperature"] if st.session_state["use_sampling"] else None),
        )
        # avoid the warning: only pass temperature when sampling
        if st.session_state["use_sampling"]:
            gen_kwargs["temperature"] = float(st.session_state["temperature"])

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **gen_kwargs,
        )

        hf_lc = HuggingFacePipeline(pipeline=hf_pipeline)
        llm = ChatHuggingFace(llm=hf_lc)

        rag_chain = build_rag_chain(vs, llm=llm, k=k, min_score=float(st.session_state["min_score"]), strict=bool(st.session_state["strict_mode"]))
        result = rag_chain.invoke({"question": q, "chat_history": st.session_state.get("chat_history", [])})

        answer_text = result["answer"].content if hasattr(result["answer"], "content") else str(result["answer"])
        sources = result["source_documents"]

        # if not sources:
        #     st.subheader("Answer")
        #     st.markdown("I don't know based on the provided sources.")
        # else:
        #     st.subheader("Answer")
        #     st.markdown(answer_text)
        st.subheader("Answer")
        st.markdown(answer_text)

        # Update history as message objects (not dicts)
        st.session_state.chat_history.append(HumanMessage(content=q))
        st.session_state.chat_history.append(AIMessage(content=answer_text))
        with st.expander("Show sources"):
            if not sources:
                st.write("No sources returned.")
            else:
                for i, d in enumerate(sources, 1):
                    path = d.metadata.get("source") or d.metadata.get("file_path") or "Unknown"
                    page = d.metadata.get("page")
                    score = d.metadata.get("relevance_score")
                    score_txt = f" — relevance {score:.3f}" if isinstance(score, (int, float)) else ""
                    st.markdown(f"**Source {i}** — `{path}`" + (f", page {page}" if page is not None else "") + score_txt)
                    st.caption(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))

        with st.expander("Chat history"):
            for m in st.session_state.chat_history[-10:]:
                role = "Human" if isinstance(m, HumanMessage) else "AI"
                st.markdown(f"**{role}:** {m.content}")
