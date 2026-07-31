import json   # read skills form json file 
import spacy  # NLP library for text processing 
from spacy.matcher import PhraseMatcher  # finds predefined skills in text 


nlp = spacy.load("en_core_web_sm")  # loads english nlp model


def load_skills(skills_path: str = "data/skills_dict.json") -> list[str]:
    """
    Load the list of skills from a JSON file.

    Args: Path to the skills JSON file.
    Returns: List of skills.
    """

    with open(skills_path) as f:  # opens the json file
        return json.load(f)   # returns lists of skills 


def build_matcher(skills: list[str]) -> PhraseMatcher:
    """
    Create a PhraseMatcher using the given skills.

    Args:  List of skills.
    Returns:  PhraseMatcher: Configured matcher for skill extraction.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # create a case-insensitive phrase matcher
    patterns = [nlp.make_doc(skill) for skill in skills] # converts plain text into a spaCy document.
    matcher.add("SKILLS",patterns) # add all skill patterns to the matcher

    return matcher 


def clean_text(text :str) -> str:
    """
    Clean the input text by removing unwanted characters and formatting.

    Args:   Raw resume text.
    Returns:   Cleaned text.
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
    Extract skills from resume text.

    Args:  Resume text.
    Returns:  Sorted list of unique skills found.
    """
    doc = nlp(text.lower())
    matches = _matcher(doc)  # finds all the matching skills from json file (skills_dict.json)
    found = {doc[start:end].text for _, start, end in matches} # store uniquely matched skills
    return sorted(found)  


# testing command for terminal :
#     uv run python -c "
# from src.parser import extract_text
# from src.preprocess import extract_skills
# text = extract_text(r'C:\Users\HP\Desktop\Resume_Screener_ATS\Drashi Adroja AIML  Resume.pdf')
# print(extract_skills(text))
# "