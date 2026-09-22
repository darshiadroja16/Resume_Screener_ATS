# AI Resume Screener + ATS Score

A resume screening tool that takes a job description and a batch of resumes, and ranks the resumes by how well they actually match — combining semantic similarity (does the resume *mean* the same thing as the JD) with keyword overlap (does it literally contain what the JD is asking for). It also tells you what's missing from each resume so you know exactly where the gap is.

I built this to actually understand how embeddings and semantic search work under the hood, instead of just calling an LLM API and calling it a day. So no GPT wrapper here — real spaCy, real Sentence-BERT, real FAISS.

## What it does

- Upload one or many resumes (PDF/DOCX) along with a job description
- Get back a ranked list with a combined ATS score for each candidate
- See exactly which required skills/keywords are missing from each resume
- Works for both use cases: screening a pile of resumes as a recruiter, or checking your own single resume against a JD

## How it works

The scoring has two parts:

1. **Semantic score** — the JD and each resume get converted into embeddings (384-number vectors) using Sentence-BERT (`all-MiniLM-L6-v2`), then compared with cosine similarity via FAISS. This catches matches even when the wording is different — "built REST APIs" and "developed backend services" score close even though they don't share exact words.
2. **Keyword score** — important terms are pulled out of the JD directly (using spaCy noun-phrase extraction, not a fixed list) and checked against the resume's raw text. This way it's not limited to some predefined skills dictionary; it adapts to whatever the JD actually asks for.

The two scores are combined into a final weighted ATS score, and whatever JD terms don't show up in a resume get listed as missing keywords.

## Tech stack

**Backend**: Python, FastAPI, spaCy, Sentence-Transformers, FAISS, scikit-learn
**Frontend**: React (Vite)
**Data prep (one-time, on Kaggle)**: pandas, Sentence-BERT, Logistic Regression

## Project structure

\`\`\`
resume-screener-ats/
├── data/
│   ├── parsed_resumes.csv       # test dataset (2,484 resumes from Kaggle)
│   └── skills_dict.json         # seed skills list, used for bulk resume tagging
├── models/
│   ├── resume_embeddings.npy    # precomputed embeddings for the test dataset
│   ├── resume_ids.npy
│   └── classifier.pkl           # resume category classifier (72% accuracy)
├── notebooks/
│   └── train_embeddings.ipynb   # the Kaggle notebook used for the one-time setup
├── src/
│   ├── parser.py                # PDF/DOCX text extraction
│   ├── preprocess.py            # skill extraction, JD term extraction
│   ├── pipeline.py              # processes newly uploaded resumes end-to-end
│   ├── ranker.py                # the actual scoring engine
│   ├── bulk_parse.py            # builds the test dataset from the Kaggle CSV
│   └── update_skills.py         # helper to grow the skills dictionary
├── frontend/                    # React app
├── main.py                      # FastAPI entrypoint
└── requirements.txt
\`\`\`

## Running it locally

You'll need Python 3.11+, Node.js, and [uv](https://docs.astral.sh/uv/) installed.

**1. Clone the repo**
\`\`\`bash
git clone https://github.com/darshiadroja16/Resume_Screener_ATS.git
cd Resume_Screener_ATS
\`\`\`

**2. Create and activate a virtual environment**
\`\`\`bash
uv venv
\`\`\`
On Windows:
\`\`\`bash
.venv\Scripts\activate
\`\`\`
On Mac/Linux:
\`\`\`bash
source .venv/bin/activate
\`\`\`

**3. Install dependencies**
\`\`\`bash
uv sync
python -m spacy download en_core_web_sm
\`\`\`

**4. Start the backend**
\`\`\`bash
uv run uvicorn main:app --reload
\`\`\`
This runs on \`http://localhost:8000\`.

**5. Set up and start the frontend** (in a separate terminal)
\`\`\`bash
cd frontend
npm install
npm run dev
\`\`\`
This runs on \`http://localhost:5173\` — open that in your browser.

**6. Use it**
Paste a job description, upload a few resumes (PDF or DOCX), and hit submit. You'll get a ranked list with scores and missing keywords for each one.

## Why this project

Mostly built to actually learn how embeddings, vector search, and the train-once/infer-many pattern work in practice — the kind of thing that shows up in real recommendation systems and search products, not just resume screeners. It's not meant to compete with actual ATS products; it's a from-scratch build to understand what's happening under the hood.

## Future Scope

- [ ] Support single-resume mode with a cleaner UI for job seekers checking their own resume
- [ ] Let recruiters adjust the semantic/keyword weight based on how strict they want matching to be
- [ ] Section-wise parsing (Education, Experience, Skills) instead of treating the whole resume as one block
- [ ] Export ranked results as CSV/PDF
- [ ] Deploy live so it's not just a localhost thing

---

built this mostly to stop guessing why my own resume wasn't clearing keyword filters 🎯

⭐ drop a star if this was useful