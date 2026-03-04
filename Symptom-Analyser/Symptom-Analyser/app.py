import streamlit as st
from rag_core import load_kb, generate_rag_answer

st.set_page_config(page_title="RAG Health Chatbot", page_icon="🩺", layout="centered")

st.title("🩺 MedlinePlus-Grounded Symptom Chatbot (RAG)")
st.caption("Educational use only. Not a diagnosis. If symptoms are severe or worsening, seek medical care.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K retrieved topics", 3, 8, 5)
    min_score = st.slider("Minimum relevance threshold", 1.0, 10.0, 4.5, 0.1)
    st.divider()
    st.write("KB file: `medical_kb_combined.csv`")

@st.cache_data
def _load():
    return load_kb("medical_kb_combined.csv")

kb = _load()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_user_msg" not in st.session_state:
    st.session_state.pending_user_msg = None

if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = None  # stores Top-K list

# Render chat history (only the messages)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User input
user_msg = st.chat_input("Type your symptoms or question...")

if user_msg:
    st.session_state.pending_user_msg = user_msg
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

# Button to generate assistant response (separate button)
btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    generate_clicked = st.button("Generate response", type="primary", disabled=st.session_state.pending_user_msg is None)
with btn_col2:
    clear_clicked = st.button("Clear chat")

if clear_clicked:
    st.session_state.messages = []
    st.session_state.pending_user_msg = None
    st.session_state.last_evidence = None
    st.rerun()

if generate_clicked and st.session_state.pending_user_msg:
    current = st.session_state.pending_user_msg
    st.session_state.pending_user_msg = None

    # history BEFORE current user message = all except the last user message we just added
    history_before = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        with st.spinner("Searching topics and generating response..."):
            answer, top, _context = generate_rag_answer(
                user_msg=current,
                history=history_before,
                kb=kb,
                top_k=top_k,
                min_score=min_score,
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_evidence = top

# Evidence button (separate, optional)
st.markdown("---")
show_ev = st.button("Show sources & evidence (Top-K)", disabled=not st.session_state.last_evidence)

if show_ev and st.session_state.last_evidence:
    with st.expander("Evidence (retrieved topics)", expanded=True):
        for i, (score, rec) in enumerate(st.session_state.last_evidence, start=1):
            st.markdown(f"**{i}. {rec.get('condition','(no title)')}** — score `{score:.2f}`")
            st.markdown(f"- Source: {rec.get('source_url','')}")
            symptoms = rec.get("common_symptoms", "")
            if symptoms:
                st.markdown(f"- Symptoms: {symptoms}")
            ov = rec.get("overview", "")
            if ov:
                st.markdown(f"- Overview: {ov[:220]}{'...' if len(ov) > 220 else ''}")
            st.markdown("")