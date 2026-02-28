"""
Scoring Module
TF-IDF Vectorization + Cosine Similarity ranking.
Uses scikit-learn as the ML backbone.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_preprocessing import clean_text
from utils.skill_extraction import extract_skills_from_text, compute_skill_overlap, parse_csv_skills


# Singleton vectorizer for fitting on the full dataset
_vectorizer = None


def get_vectorizer():
    """Return or build the TF-IDF vectorizer."""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),      # Capture bigrams like "machine learning"
            max_features=8000,        # Vocabulary cap for performance
            min_df=1,                 # Include rare terms (important for skills)
            sublinear_tf=True,        # Apply log normalization
            strip_accents='unicode',
            analyzer='word',
        )
    return _vectorizer


def fit_vectorizer_on_dataset(df, text_col='resume_text'):
    """Fit the TF-IDF vectorizer on the full CSV dataset for better vocabulary."""
    global _vectorizer
    _vectorizer = get_vectorizer()
    corpus = df[text_col].dropna().astype(str).apply(clean_text).tolist()
    if corpus:
        _vectorizer.fit(corpus)
    return _vectorizer


def compute_similarity(jd_text, candidate_text):
    """
    Compute cosine similarity between job description and candidate text.
    Returns score as a percentage (0–100).
    """
    vectorizer = get_vectorizer()
    cleaned_jd = clean_text(jd_text)
    cleaned_candidate = clean_text(candidate_text)

    try:
        # Fit on both if vectorizer not yet fitted
        if not hasattr(vectorizer, 'vocabulary_'):
            vectorizer.fit([cleaned_jd, cleaned_candidate])

        jd_vec = vectorizer.transform([cleaned_jd])
        candidate_vec = vectorizer.transform([cleaned_candidate])
        score = cosine_similarity(jd_vec, candidate_vec)[0][0]
    except Exception:
        score = 0.0

    return round(float(score) * 100, 2)


def categorize_fit(score):
    """Categorize candidate based on similarity score."""
    if score >= 75:
        return 'Strong Fit', 'strong'
    elif score >= 50:
        return 'Moderate Fit', 'moderate'
    else:
        return 'Weak Fit', 'weak'


def rank_candidates_from_csv(df, job_description, sample_size=100):
    """
    Rank candidates from the CSV dataset against a job description.
    Returns a list of ranked candidate dicts.
    """
    # Build resume text from available columns
    def build_resume_text(row):
        parts = []
        for col in ['career_objective', 'skills', 'responsibilities',
                    'positions', 'major_field_of_studies', 'certification_skills',
                    'related_skils_in_job']:
            val = row.get(col, '')
            if val and str(val) not in ['nan', 'None', '[]', '[None]']:
                parts.append(str(val))
        return ' '.join(parts)

    df = df.copy()
    df['resume_text'] = df.apply(build_resume_text, axis=1)

    # Sample for performance (use stratified if large)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Fit vectorizer on this dataset + JD
    all_texts = df['resume_text'].apply(clean_text).tolist()
    all_texts.append(clean_text(job_description))

    vectorizer = get_vectorizer()
    vectorizer.fit(all_texts)

    # Extract JD skills
    jd_skills = extract_skills_from_text(job_description)

    results = []
    for idx, row in df.iterrows():
        resume_text = row['resume_text']
        score = compute_similarity(job_description, resume_text)

        # Skills from CSV column (pre-extracted)
        candidate_skills = parse_csv_skills(str(row.get('skills', '')))
        if not candidate_skills:
            candidate_skills = extract_skills_from_text(resume_text)

        matched, missing = compute_skill_overlap(candidate_skills, jd_skills)
        fit_label, fit_class = categorize_fit(score)

        # Build candidate name from position + index
        position = str(row.get('positions', '')).replace("['", '').replace("']", '').strip()
        name = f"Candidate #{idx + 1}" + (f" ({position})" if position and position != 'nan' else '')

        results.append({
            'name': name,
            'score': score,
            'fit_label': fit_label,
            'fit_class': fit_class,
            'matched_skills': matched[:15],   # Cap for display
            'missing_skills': missing[:15],
            'all_skills': candidate_skills[:20],
            'career_objective': str(row.get('career_objective', ''))[:300],
            'source': 'csv',
        })

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1

    return results


def rank_candidates_from_uploads(files_data, job_description):
    """
    Rank candidates from uploaded file data.
    files_data: list of dicts with 'name' and 'text' keys.
    Returns ranked list of candidate dicts.
    """
    # Fit vectorizer on JD + all uploaded resumes
    all_texts = [clean_text(job_description)] + [clean_text(f['text']) for f in files_data]
    vectorizer = get_vectorizer()
    vectorizer.fit(all_texts)

    jd_skills = extract_skills_from_text(job_description)

    results = []
    for f in files_data:
        resume_text = f['text']
        score = compute_similarity(job_description, resume_text)

        candidate_skills = extract_skills_from_text(resume_text)
        matched, missing = compute_skill_overlap(candidate_skills, jd_skills)
        fit_label, fit_class = categorize_fit(score)

        results.append({
            'name': f['name'],
            'score': score,
            'fit_label': fit_label,
            'fit_class': fit_class,
            'matched_skills': matched[:15],
            'missing_skills': missing[:15],
            'all_skills': candidate_skills[:20],
            'career_objective': '',
            'source': 'upload',
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1

    return results
