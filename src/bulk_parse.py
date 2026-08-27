import pandas as pd
from src.preprocess import extract_skills


def build_dataset_from_csv(csv_path: str, text_column: str, id_column: str) -> pd.DataFrame:
    """
    Build a structured DataFrame from a CSV file by extracting resume text and skills using
    the specified text and ID columns.
    """
    df = pd.read_csv(csv_path)
    rows = []  # store processed resumes

    for _, row in df.iterrows():  # process each resume from the csv file 
        text = row[text_column]  # get resume text
        if pd.isna(text): # skip empty resumes
            continue
        rows.append({
            "id": row[id_column],
            "raw_text": text,
            "skills": extract_skills(text),
        })

    return pd.DataFrame(rows) # returns processed resume information into dataframe 