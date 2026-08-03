import pandas as pd
from src.preprocess import extract_skills


def build_dataset_from_csv(csv_path: str, text_column: str, id_column: str) -> pd.DataFrame:
    """
    Build a structured dataset from a CSV file containing resumes

    Args:
        - Path to the input CSV file
        - Col containing resume text in input csv
        - Col containing unique resume IDs in input csv
    Returns:
        DataFrame containing:
            - id
            - raw_text
            - skills
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

# testing code
if __name__ == "__main__":
    dataset = build_dataset_from_csv("data/Resume.csv", text_column="Resume_str", id_column="ID")
    dataset.to_csv("data/parsed_resumes.csv", index=False)
    print(f"Saved {len(dataset)} resumes to data/parsed_resumes.csv")