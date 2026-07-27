"""
Paytech HR Helpdesk — Streamlit version
-----------------------------------------
A free, retrieval-grounded HR chatbot. Answers are generated only from the
documents in knowledge_base/ (Paytech's Employee Handbook + Employee
Relations & Terminations SOP). Deploys for free on Streamlit Community
Cloud and calls Groq's free LLM API for generation.

To add or update policy documents: drop new .txt files into knowledge_base/,
commit to GitHub, and Streamlit Cloud redeploys automatically.
"""

import os
import glob
import urllib.parse
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
GROQ_MODEL = "llama-3.3-70b-versatile"   # free tier on Groq
TOP_K = 5                                 # chunks retrieved per question
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40

# Friendly display names for source documents (falls back to filename if not listed)
SOURCE_DISPLAY_NAMES = {
    "Paytech_Handbook_Policy_Descriptive.txt": "Employee Handbook — Volume 6 (Company Policies)",
    "Paytech_SOP_Employee_Relations_Terminations.txt": "SOP ER-TERM-001 — Employee Relations & Terminations",
    "Paytech_Handbook_Volume7_Additional_Policies.txt": "Employee Handbook — Volume 7 (Additional Policies)",
}

# Where "escalate to HR" emails should be addressed. Override by adding
# HR_ESCALATION_EMAIL to this app's Secrets on Streamlit Cloud.
DEFAULT_ESCALATION_EMAIL = "hr@paytech.example"

SYSTEM_PROMPT = """You are the Paytech HR Helpdesk assistant.

Rules you must always follow:
1. Answer ONLY using the information given to you in the "Context" section below. Do not use outside knowledge about HR law or general practice.
2. If the context does not contain enough information to answer confidently, say so plainly and tell the employee to contact the People & Culture (HR) team directly. Never guess or make up a policy detail.
3. Be concise, warm, and professional — like a helpful HR generalist, not a legal document.
4. When relevant, mention which policy/section the answer comes from (e.g. "Per the Termination policy, Section 13...").
5. Do not give legal advice. For anything involving discipline, termination, harassment, investigations, or a specific employee's situation, remind the user that Employee Relations / Compliance & Legal must be looped in per SOP ER-TERM-001.
6. Keep answers under ~150 words unless the question genuinely requires more detail.
"""

# ---------------------------------------------------------------------------
# Knowledge base loading + chunking (cached so it only runs once per deploy)
# ---------------------------------------------------------------------------

def load_documents():
    docs = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


@st.cache_resource(show_spinner="Loading HR knowledge base...")
def build_index():
    docs = load_documents()
    all_chunks, metadata = [], []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append(chunk)
            metadata.append(doc["source"])

    if not all_chunks:
        return None, None, None, []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(all_chunks)
    return vectorizer, matrix, all_chunks, metadata


VECTORIZER, MATRIX, CHUNKS, SOURCES = build_index()


def retrieve(query, top_k=TOP_K):
    if VECTORIZER is None:
        return []
    q_vec = VECTORIZER.transform([query])
    sims = cosine_similarity(q_vec, MATRIX).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        if sims[i] > 0:
            results.append({"text": CHUNKS[i], "source": SOURCES[i], "score": float(sims[i])})
    return results


# ---------------------------------------------------------------------------
# LLM call (Groq — free tier)
# ---------------------------------------------------------------------------

def get_api_key():
    # Streamlit Cloud secrets first, then environment variable (for local runs)
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


def get_escalation_email():
    try:
        return st.secrets["HR_ESCALATION_EMAIL"]
    except Exception:
        return os.environ.get("HR_ESCALATION_EMAIL", DEFAULT_ESCALATION_EMAIL)


def answer_question(message, history):
    """Returns (reply_text, sources_used) so the UI can show citations."""
    api_key = get_api_key()
    if not api_key:
        return (
            "⚠️ The AI backend isn't configured yet. An admin needs to add "
            "GROQ_API_KEY under this app's Settings → Secrets on Streamlit "
            "Community Cloud. Get a free key at console.groq.com.",
            [],
        )

    client = Groq(api_key=api_key)

    hits = retrieve(message)
    if not hits:
        context = "No matching policy content was found."
    else:
        context = "\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-4:]:
        # only pass role/content to the LLM, not our extra "sources" key
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nEmployee question: {message}",
    })

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        return completion.choices[0].message.content, hits
    except Exception as e:
        return f"⚠️ Something went wrong reaching the AI service: {e}", hits


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Paytech HR Helpdesk", page_icon="🧑‍💼")

st.title("🧑‍💼 Paytech HR Helpdesk")
st.caption(
    "Ask about leave policies, code of conduct, discipline, terminations, and "
    "other HR topics covered in the Employee Handbook and related SOPs. This "
    "assistant only answers from Paytech's official HR documents — for "
    "anything else, please contact People & Culture (HR) directly."
)

def render_sources_and_escalation(sources, question, answer, key_suffix):
    """Shows which handbook sections were used, and an escalate-to-HR link."""
    if sources:
        with st.expander(f"📄 Sources ({len(sources)})"):
            seen = set()
            for h in sources:
                label = SOURCE_DISPLAY_NAMES.get(h["source"], h["source"])
                if label in seen:
                    continue
                seen.add(label)
                snippet = h["text"][:220].strip()
                st.markdown(f"**{label}**\n\n> {snippet}…")

    escalation_email = get_escalation_email()
    subject = urllib.parse.quote("HR Helpdesk escalation")
    answer_snippet = answer if len(answer) <= 500 else answer[:500] + "…"
    body = urllib.parse.quote(
        f"Original question: {question}\n\n"
        f"HR Helpdesk's answer: {answer_snippet}\n\n"
        f"(Escalated from the Paytech HR Helpdesk — please follow up directly.)"
    )
    mailto = f"mailto:{escalation_email}?subject={subject}&body={body}"
    st.markdown(
        f"<a href='{mailto}' style='font-size:0.85em;'>✉️ Not quite what you needed? Escalate to HR</a>",
        unsafe_allow_html=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            prev_question = (
                st.session_state.messages[i - 1]["content"] if i > 0 else ""
            )
            render_sources_and_escalation(
                msg.get("sources", []), prev_question, msg["content"], key_suffix=i
            )

example_cols = st.columns(2)
examples = [
    "How much vacation time do I get?",
    "What's the process if I want to raise a harassment complaint?",
    "What happens during a termination meeting?",
    "Can I work remotely?",
]
if not st.session_state.messages:
    st.write("Try asking:")
    for i, ex in enumerate(examples):
        if example_cols[i % 2].button(ex, use_container_width=True):
            st.session_state.pending_prompt = ex

prompt = st.chat_input("Ask an HR question...")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking HR policies..."):
            reply, sources = answer_question(prompt, st.session_state.messages[:-1])
        st.markdown(reply)
        render_sources_and_escalation(sources, prompt, reply, key_suffix="latest")

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
