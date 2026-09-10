import json   # read skills form json file 
import spacy  # NLP library for text processing 
from spacy.matcher import PhraseMatcher  # finds predefined skills in text 

nlp_full = spacy.load("en_core_web_sm")  # parser enabled(sirf jd ke liye)
nlp = spacy.load("en_core_web_sm",disable=["parser", "ner", "lemmatizer", "tagger"])  # loads english nlp model

GENERIC_TERMS = {"experience", "team", "company", "candidate", "role", "years",
                  "knowledge", "skills", "work", "opportunity", "job", "position"}

def load_skills(skills_path: str = "data/skills_dict.json") -> list[str]:
    """ 
    Load the list of skills from a JSON file using the given file path. 
    """

    with open(skills_path) as f:  # opens the json file
        return json.load(f)   # returns lists of skills 


def build_matcher(skills: list[str]) -> PhraseMatcher:
    """ 
    Create a case-insensitive PhraseMatcher using the given list of skills. 
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # create a case-insensitive phrase matcher
    patterns = [nlp.make_doc(skill) for skill in skills] # converts plain text into a spaCy document.
    matcher.add("SKILLS",patterns) # add all skill patterns to the matcher

    return matcher 


def clean_text(text :str) -> str:
    """ 
    Clean resume text by removing stop words and punctuation and
    converting words to lowercase lemmas. 
    """
    doc = nlp(text)  # process the text using spacy
    tokens = [
    t.lemma_.lower() # lemma_ convert words to there normal form like running to run 
    for t in doc 
    if not t.is_stop and not t.is_punct  # to remove stop words and puntuations
    ]
    return " ".join(tokens)


_skills = load_skills()   # loads the skills once when the peogram starts 
_matcher = build_matcher(_skills)  # builds the matcher once when the program starts


def extract_skills(text: str) -> list[str]:
    """ 
    Extract unique skills from resume text using the predefined skill matcher
    and return them in sorted order. 
    """
    doc = nlp(text.lower())
    matches = _matcher(doc)  # finds all the matching skills from json file (skills_dict.json)
    found = {doc[start:end].text for _, start, end in matches} # store uniquely matched skills
    return sorted(found)  

def extract_jd_terms(jd_text: str) -> set[str]:
    """
    Pulls out candidate skill/requirement phrases directly from the JD —
    both known dictionary skills AND noun phrases not yet in the dictionary.
    """
    known = set(extract_skills(jd_text))
    doc = nlp_full(jd_text.lower())
    candidates = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase) <= 2:
            continue
        if phrase in GENERIC_TERMS:
            continue
        candidates.add(phrase)
 
    return known | candidates
 
 
def terms_present_in_text(terms: set[str], text: str) -> set[str]:
    """
    Direct substring check — doesn't rely on the skills dictionary at all.
    """
    text_lower = text.lower()
    return {term for term in terms if term in text_lower}