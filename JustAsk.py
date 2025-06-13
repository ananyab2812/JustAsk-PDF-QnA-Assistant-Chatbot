import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# -------------------- CONFIG -------------------- #
st.set_page_config(page_title="JustAsk – PDF Chatbot", layout="wide")

# -------------------- STYLING + VOICE SCRIPT -------------------- #
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #1c1c1e;
    color: white;
}
.chat-bubble {
    background-color: #2a2a2a;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 12px;
    font-size: 0.95rem;
    color: white;
}
.user-bubble {
    background-color: #373742;
    text-align: right;
}
.bot-bubble {
    background-color: #2a2a2a;
}
.bottom-bar {
    position: fixed;
    bottom: 1rem;
    left: 20%;
    width: 60%;
    background-color: #1c1c1e;
    border-radius: 1rem;
    padding: 1rem;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
}
</style>

<script>
function startDictation() {
    var recognition = new(window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = function(event) {
        var transcript = event.results[0][0].transcript;
        document.querySelector('input[type="text"]').value = transcript;
        document.querySelector('button[type="submit"]').click();
    };
    recognition.start();
}
</script>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR -------------------- #
with st.sidebar:
    st.title("Chat History")
    if "history" in st.session_state and st.session_state.history:
        for idx, (q, _) in enumerate(st.session_state.history):
            st.markdown(f"- Q{idx+1}: {q}")
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()

# -------------------- SESSION STATE INIT -------------------- #
if "history" not in st.session_state:
    st.session_state.history = []
if "qa" not in st.session_state:
    st.session_state.qa = None

# -------------------- HEADER -------------------- #
st.markdown("<h2 style='text-align: center;'>JustAsk – PDF Q&A Assistant</h2>", unsafe_allow_html=True)

# -------------------- CHAT DISPLAY -------------------- #
chat_container = st.container()
with chat_container:
    for user_msg, bot_msg in st.session_state.history:
        st.markdown(f"<div class='chat-bubble user-bubble'>{user_msg}</div>", unsafe_allow_html=True)
        st.markdown(bot_msg, unsafe_allow_html=True)

# -------------------- BOTTOM BAR -------------------- #
with st.container():
    st.markdown("<div class='bottom-bar'>", unsafe_allow_html=True)
    cols = st.columns([3, 1])
    with cols[0]:
        with st.form(key="chat_input_form", clear_on_submit=True):
            col_input, col_voice = st.columns([10, 1])
            with col_input:
                user_query = st.text_input("Type your question", label_visibility="collapsed")
            with col_voice:
                st.markdown('<button onclick="startDictation()" style="margin-top:6px;">🎤</button>', unsafe_allow_html=True)
            submit = st.form_submit_button("Send")
    with cols[1]:
        pdf_upload = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- LLM PIPELINE -------------------- #
@st.cache_resource
def load_llm_pipeline():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
    return HuggingFacePipeline(pipeline=pipe)

# -------------------- ANSWER FORMATTER -------------------- #
def format_answer(text):
    lines = text.strip().split("\n")
    formatted = "<div class='chat-bubble bot-bubble'>"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            formatted += f"<b>{key.strip()}:</b> {val.strip()}<br>"
        elif line.startswith("-") or line.startswith("•"):
            formatted += f"<li>{line[1:].strip()}</li>"
        else:
            formatted += f"{line}<br>"
    formatted += "</div>"
    return formatted

# -------------------- PDF UPLOAD + PROCESS -------------------- #
if pdf_upload:
    with st.spinner("Processing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_upload.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()[:5]  # Read only first 5 pages
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        llm = load_llm_pipeline()
        st.session_state.qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        st.success("PDF uploaded and processed successfully!")

# -------------------- CHAT LOGIC -------------------- #
if submit and user_query and st.session_state.qa:
    with st.spinner("Thinking..."):
        raw_answer = st.session_state.qa.run(user_query)
        beautified = format_answer(raw_answer)
        st.session_state.history.append((user_query, beautified))
        st.markdown(f"""
            <script>
            var msg = new SpeechSynthesisUtterance("{raw_answer.replace('"', '').replace("'", '')}");
            msg.lang = "en-IN";
            window.speechSynthesis.speak(msg);
            </script>
        """, unsafe_allow_html=True)
        st.rerun()
