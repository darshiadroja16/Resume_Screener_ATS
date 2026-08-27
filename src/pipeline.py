import numpy as np
from pathlib import Path
from src.parser import extract_text_safe
from src.preprocess import extract_skills
from src.ranker import embed_text


def process_uploaded_resumes(folder_path: str):
    """ 
    Extract text, skills, and embeddings from PDF/DOCX resumes in a folder. 
    """
    ids, texts, skills_list, embeddings = [], [], [], [] 

    # iterate through each resume from the specified folder(sample resume folder)
    for file in Path(folder_path).iterdir():

        if file.suffix.lower() not in (".pdf", ".docx"):
            continue
        text = extract_text_safe(str(file))
        if text is None:
            continue
        # store the filename without its extension as the resume id
        ids.append(file.stem)
        texts.append(text)
        skills_list.append(extract_skills(text))
        embeddings.append(embed_text(text))

    # return all processed resume info in a structured dictionary format
    return {
        "ids": np.array(ids),
        "texts": texts,
        "skills": skills_list,
        "embeddings": np.array(embeddings),
    }