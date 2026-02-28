"""
Text Preprocessing Pipeline
Handles lowercasing, stopword removal, lemmatization, and special character removal.
Implements a spaCy-compatible NLP pipeline using pure Python + sklearn.
"""

import re
import string

# Comprehensive English stopwords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'she', 'it', 'they', 'them', 'this', 'that',
    'these', 'those', 'which', 'who', 'whom', 'what', 'when', 'where', 'how',
    'why', 'not', 'no', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'as', 'if', 'then', 'than', 'too', 'very', 'just', 'also', 'more',
    'other', 'any', 'some', 'such', 'own', 'same', 'few', 'most', 'each',
    'all', 'both', 'over', 'under', 'again', 'further', 'once', 'here',
    'there', 'only', 'own', 'their', 'his', 'her', 'its', 'get', 'got',
    'us', 'him', 'she', 'its', 'than', 'between', 'among', 'while', 'since',
    'before', 'after', 'above', 'below', 'off', 'out', 'because', 'although',
    'however', 'within', 'without', 'across', 'along', 'around', 'behind',
    'beside', 'besides', 'beyond', 'despite', 'except', 'inside', 'outside',
    'per', 'regarding', 'until', 'upon', 'versus', 'via', 'within', 'throughout'
}

# Basic lemmatization rules (suffix stripping)
LEMMA_RULES = [
    (r'ings$', 'ing'), (r'ations$', 'ation'), (r'ities$', 'ity'),
    (r'ments$', 'ment'), (r'nesses$', 'ness'), (r'ists$', 'ist'),
    (r'izers$', 'izer'), (r'izers$', 'ize'), (r'ers$', 'er'),
    (r'ies$', 'y'), (r'es$', ''), (r'ed$', ''), (r'ly$', ''),
    (r'ing$', ''), (r's$', ''),
]


def lemmatize(word):
    """Simple rule-based lemmatizer."""
    if len(word) <= 3:
        return word
    for pattern, replacement in LEMMA_RULES:
        new_word = re.sub(pattern, replacement, word)
        if new_word != word and len(new_word) >= 3:
            return new_word
    return word


def clean_text(text):
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove special characters & numbers
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    Returns cleaned string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Preserve hyphenated words and dots in acronyms (C++, .NET)
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords, short tokens, and pure numbers
    tokens = [
        t for t in tokens
        if t not in STOPWORDS
        and len(t) > 2
        and not t.isdigit()
    ]

    # Lemmatize
    tokens = [lemmatize(t) for t in tokens]

    return ' '.join(tokens)


def extract_raw_text_tokens(text):
    """Extract raw tokens preserving case for skill matching."""
    if not text or not isinstance(text, str):
        return []
    tokens = re.findall(r'\b[\w\+\#\.]+\b', text)
    return [t for t in tokens if len(t) > 1]
