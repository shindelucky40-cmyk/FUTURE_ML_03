# RecruitAI — Resume Screening & Ranking System


<img width="1888" height="910" alt="Screenshot 2026-02-28 102840" src="https://github.com/user-attachments/assets/48f7d34d-fd2d-46da-8db3-596345f23d8a" />


<p align="center">
  An ML-powered web app that automatically scores, ranks, and analyzes candidates against any job description — using TF-IDF vectorization, cosine similarity, and a dynamic NLP skill extraction pipeline.
</p>

---

## Overview

Manually reviewing resumes is slow, inconsistent, and doesn't scale. RecruitAI solves this by converting unstructured resume text into machine-comparable vectors, ranking every candidate by relevance to your job description, and surfacing exactly which skills each candidate has — and which they're missing.

**Two screening modes:**
- **Dataset Mode** — Screen from a preloaded dataset of 9,544 real-world resumes with optional role filtering
- **Upload Mode** — Upload your own PDF, DOCX, or TXT resumes for instant scoring

---

## Features

- Paste any job description and rank candidates in seconds
- TF-IDF + cosine similarity scoring (0–100%)
- Dynamic NLP skill extraction — no predefined skill list required
- Matched vs. missing skill gap analysis per candidate
- Strong / Moderate / Weak Fit categorization
- Side-by-side candidate comparison (up to 4)
- Filter by fit category, search by name or skill
- Export results to CSV or PDF
- Dark terminal UI with score distribution bar

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML / NLP | scikit-learn (TF-IDF, cosine similarity) |
| Text Extraction | pdfminer.six, python-docx |
| PDF Export | ReportLab |
| Frontend | HTML, CSS, Vanilla JS |
| Dataset | 9,544-record resume CSV |

---

## How It Works

### 1 — Text Preprocessing
Each resume and the job description go through a cleaning pipeline: lowercasing → special character removal → stopword filtering → rule-based lemmatization.

### 2 — TF-IDF Vectorization
The vectorizer is fitted on the full candidate corpus plus the job description together, building a shared vocabulary. Each document becomes a weighted term-frequency vector. Bigrams (`ngram_range=(1,2)`) capture multi-word skills like "machine learning" and "REST API" as single units. Sublinear TF normalization (`sublinear_tf=True`) prevents repetition from inflating scores.

**Why TF-IDF over embeddings?** TF-IDF is unsupervised, requires no pretrained model, and is fully interpretable. Common resume filler words are automatically down-weighted; rare, specific technical terms that appear in both the JD and a resume get boosted. For an internship-level project where explainability matters, it's the right tradeoff.

### 3 — Cosine Similarity Scoring

```
score (%) = cosine_similarity(JD vector, candidate vector) × 100
```

Cosine similarity measures the angle between two vectors regardless of their magnitude, making it document-length invariant. A 1-page and a 5-page resume are compared fairly. The output maps directly to a 0–100% relevance score.

### 4 — Skill Extraction (NLP Pipeline)
A three-stage extractor inspired by spaCy's noun chunk pipeline:

1. **Multi-word phrase matching** — regex patterns for 30+ common technical phrases (`machine learning`, `Spring Boot`, `REST API`, etc.)
2. **Proper noun heuristics** — sequences of capitalized tokens flagged as tools/products (`TensorFlow`, `AWS`, `PostgreSQL`)
3. **Technical token detection** — tokens with known tech suffixes (`sql`, `js`, `db`) or prefixes (`mongo`, `redis`, `elastic`)

Skills are normalized, deduplicated case-insensitively, and compared against JD skills using substring matching to produce matched and missing skill lists per candidate.

> The pipeline is designed to drop in spaCy `en_core_web_sm` for production-grade NER — see `utils/skill_extraction.py`.

### 5 — Ranking & Categorization

| Score | Category |
|---|---|
| ≥ 75% | ✅ Strong Fit |
| 50 – 74% | 🟡 Moderate Fit |
| < 50% | 🔴 Weak Fit |

---

## Project Structure

```
resume-screening-system/
│
├── app.py                        # Flask app — routes, file parsing, export
├── utils/
│   ├── text_preprocessing.py     # Cleaning, stopwords, lemmatization
│   ├── skill_extraction.py       # NLP pipeline + skill gap analysis
│   └── scoring.py                # TF-IDF vectorizer + cosine similarity
│
├── templates/
│   └── index.html                # Single-page UI
│
├── static/
│   └── style.css                 # Dark theme stylesheet
│
├── dataset/
│   └── resumes.csv               # 9,544-record resume dataset
│
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/shindelucky40-cmyk/
FUTURE_ML_03
cd
FUTURE_ML_03

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

**Optional — upgrade to spaCy NLP:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
# Then enable it in utils/skill_extraction.py
```

---

## Limitations

- **Lexical matching only.** TF-IDF doesn't understand that "ML" and "machine learning" mean the same thing. Bigrams mitigate this but don't fully solve it.
- **Heuristic skill extraction.** The NLP pipeline uses regex patterns. Unusual skill names may be missed; spaCy NER in production reduces false negatives.
- **No resume structure awareness.** The system treats the full resume as a bag of words — it doesn't distinguish a "Skills" section from an "Interests" section.
- **Scanned PDFs unsupported.** Image-based PDFs cannot be parsed. Text-based PDFs only.
- **Score is relative, not absolute.** A candidate who mirrors JD language scores higher than an equally qualified candidate who doesn't. This is a known TF-IDF limitation.
- **CSV mode uses sampling.** Up to 200 candidates are scored per query for performance. Results may vary slightly across runs.

---

## Future Improvements

- [ ] BERT / Sentence-Transformers for semantic similarity (`"NLP" ↔ "natural language processing"`)
- [ ] Resume section parsing to weight Skills and Experience sections separately
- [ ] Recruiter feedback loop to fine-tune scoring weights over time
- [ ] Candidate clustering by skill profile (K-Means)
- [ ] Async batch processing with Celery + Redis for 500+ resumes
- [ ] ATS integration (Greenhouse, Lever, Workday)

---
## Author
     lalit shinde

Built as an ML Internship Portfolio Project
