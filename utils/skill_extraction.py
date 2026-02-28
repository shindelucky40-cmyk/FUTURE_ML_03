"""
Skill Extraction Module
Implements a dynamic NLP pipeline that mimics spaCy's approach:
- Tokenization → POS-like heuristics → Noun/technical entity extraction
- Works without spaCy by using regex patterns + linguistic rules
- Designed to be swapped with spaCy en_core_web_sm for production use
"""

import re
from utils.text_preprocessing import STOPWORDS

# Common technical suffixes that indicate skills/tools
TECH_INDICATORS = {
    'suffixes': ['js', 'py', 'db', 'sql', 'ml', 'ai', 'api', 'sdk', 'os',
                 'net', 'ui', 'ux', 'ci', 'cd', 'oop', 'orm'],
    'prefixes': ['react', 'node', 'mongo', 'redis', 'post', 'my', 'elastic',
                 'dynamo', 'angular', 'vue', 'spring', 'django', 'flask', 'fast'],
}

# Patterns for multi-word technical skills
MULTIWORD_PATTERNS = [
    r'\b(?:machine\s+learning|deep\s+learning|natural\s+language\s+processing|'
    r'data\s+science|data\s+engineering|data\s+analysis|big\s+data|'
    r'computer\s+vision|neural\s+network|reinforcement\s+learning|'
    r'transfer\s+learning|feature\s+engineering|time\s+series|'
    r'web\s+development|mobile\s+development|full\s+stack|front\s+end|back\s+end|'
    r'cloud\s+computing|devops|version\s+control|agile\s+methodology|'
    r'object\s+oriented|test\s+driven|continuous\s+integration|'
    r'restful\s+api|rest\s+api|graphql\s+api|micro\s+services|'
    r'amazon\s+web\s+services|google\s+cloud|microsoft\s+azure|'
    r'power\s+bi|tableau\s+server|apache\s+spark|apache\s+kafka|'
    r'spring\s+boot|node\s+js|react\s+native|next\s+js|nuxt\s+js|'
    r'c\s*\+\+|\.net\s+core|asp\.net|entity\s+framework|'
    r'scikit\s+learn|tensor\s+flow|pytorch|keras)\b',
]

# Single-word technical skills (capitalized = likely a proper noun/tool)
PROPER_NOUN_MIN_LEN = 2
NUMERIC_TOKEN_PATTERN = re.compile(r'^\d+(\.\d+)?$')


def extract_noun_chunks(text):
    """
    Extract noun-like phrases using linguistic heuristics.
    Simulates spaCy's noun_chunks pipeline.
    """
    chunks = set()

    # 1. Extract multi-word technical phrases
    text_lower = text.lower()
    for pattern in MULTIWORD_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for m in matches:
            cleaned = ' '.join(m.strip().split())
            if cleaned:
                chunks.add(cleaned.title())

    # 2. Extract capitalized sequences (proper noun heuristic)
    # Sequences of 1-3 capitalized words = technical names / product names
    cap_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9\+\#\.]*(?:\s+[A-Z][a-zA-Z0-9\+\#\.]+){0,2})\b')
    for match in cap_pattern.finditer(text):
        phrase = match.group(1).strip()
        words = phrase.split()
        # Filter out sentence-starting false positives
        if all(w not in [w.lower() for w in STOPWORDS] for w in words):
            if len(phrase) >= 2 and not NUMERIC_TOKEN_PATTERN.match(phrase):
                chunks.add(phrase)

    # 3. Extract tokens with technical suffixes
    tokens = re.findall(r'\b[\w\+\#\.]+\b', text)
    for token in tokens:
        tl = token.lower()
        if any(tl.endswith(s) for s in TECH_INDICATORS['suffixes']) or \
           any(tl.startswith(p) for p in TECH_INDICATORS['prefixes']):
            if len(token) >= 2:
                chunks.add(token)

    return chunks


def extract_skills_from_text(text):
    """
    Main skill extraction function.
    Returns a sorted list of extracted skills from raw text.
    """
    if not text or not isinstance(text, str):
        return []

    skills = set()

    # Extract noun chunks (NLP-based)
    skills.update(extract_noun_chunks(text))

    # Filter noise: remove pure stopwords, single chars, numbers
    filtered = []
    for skill in skills:
        words = skill.lower().split()
        # Skip if all words are stopwords
        if all(w in STOPWORDS for w in words):
            continue
        # Skip very short or purely numeric
        if len(skill) < 2 or NUMERIC_TOKEN_PATTERN.match(skill):
            continue
        # Skip common non-skill words
        if skill.lower() in {'the', 'with', 'and', 'for', 'have', 'from',
                              'that', 'this', 'been', 'will', 'also', 'they'}:
            continue
        filtered.append(skill)

    # Deduplicate (case-insensitive) keeping the best formatted version
    seen = {}
    for skill in filtered:
        key = skill.lower()
        if key not in seen:
            seen[key] = skill
        else:
            # Prefer mixed case over all-caps or all-lower
            existing = seen[key]
            if skill.istitle() or (not skill.isupper() and not skill.islower()):
                seen[key] = skill

    return sorted(seen.values())


def compute_skill_overlap(candidate_skills, jd_skills):
    """
    Compare candidate skills vs JD skills.
    Returns (matched_skills, missing_skills) as lists.
    """
    # Normalize to lowercase sets for comparison
    candidate_set = {s.lower().strip() for s in candidate_skills}
    jd_set = {s.lower().strip() for s in jd_skills}

    # Fuzzy-ish matching: check if jd skill appears in candidate skill or vice versa
    matched = []
    missing = []

    for jd_skill in jd_set:
        # Direct match or substring match
        found = False
        for c_skill in candidate_set:
            if jd_skill == c_skill or jd_skill in c_skill or c_skill in jd_skill:
                found = True
                break
        if found:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    return sorted(matched), sorted(missing)


def parse_csv_skills(skills_str):
    """Parse the skills column from CSV dataset (stored as Python list string)."""
    if not skills_str or not isinstance(skills_str, str):
        return []
    # Remove brackets and quotes
    cleaned = re.sub(r"[\[\]'\"\\]", '', str(skills_str))
    skills = [s.strip() for s in cleaned.split(',') if s.strip()]
    return skills
