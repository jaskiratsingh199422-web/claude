# Paytech HR Helpdesk

A free, retrieval-grounded AI HR helpdesk. It answers employee questions
strictly from the documents in `knowledge_base/`:

- `Paytech_Handbook_Policy_Descriptive.txt` — Employee Handbook, Volume 6 (Company Policies)
- `Paytech_SOP_Employee_Relations_Terminations.txt` — Employee Relations & Terminations SOP (ER-TERM-001)

Deployed as a Streamlit app (`streamlit_app.py`) on **Streamlit Community
Cloud** — free, no payment method required.

## How it works

1. On startup, both documents are split into overlapping ~180-word chunks.
2. A TF-IDF index is built over the chunks (no paid embedding API needed).
3. When an employee asks a question, the top 5 most relevant chunks are
   retrieved and passed as context to a free Groq-hosted LLM
   (`llama-3.3-70b-versatile`), along with strict instructions to answer only
   from that context.
4. If nothing relevant is found, the bot tells the employee to contact HR
   instead of guessing.

## Setup (one-time)

1. Create a free account at [console.groq.com](https://console.groq.com) and
   generate an API key (no credit card required).
2. On Streamlit Community Cloud, open this app's **Settings → Secrets** and
   add:
   ```
   GROQ_API_KEY = "your-key-here"
   ```
3. The app redeploys automatically and is live at its public
   `*.streamlit.app` URL.

## Updating the knowledge base

Replace or add `.txt` files in `knowledge_base/` in the GitHub repo, commit,
and Streamlit Cloud redeploys automatically. Keep documents in plain text
for best retrieval results.

## Cost

- Streamlit Community Cloud: $0
- Groq API free tier: $0 (rate-limited, generous for internal HR helpdesk use)

## Note

`app.py` (Gradio version) is also included in case you later want to deploy
on Hugging Face Spaces PRO or another Gradio-compatible host — it is not
used by the Streamlit Cloud deployment.
