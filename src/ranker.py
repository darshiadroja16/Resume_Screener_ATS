import ast
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.preprocess import extract_jd_terms, terms_present_in_text

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> np.ndarray:
    """
    Convert resume text into a np.ndarray: Embedding vector.
    """
    return _model.encode([text])[0]


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index from np.ndarray of resume embeddings.
    """
    # normalization makes embedding's length/norm  1 (length/norm is the square root of the sum of squares of all elements in the vector)
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # embedding is diveded by norm to make it normalized  
    index = faiss.IndexFlatIP(normalized.shape[1]) # creates the faiss index using inner-product(IP) similarity search method.shape[1] gives dimension of embedding
    index.add(normalized) 
    return index


def rank_by_similarity(jd_text: str, embeddings: np.ndarray, resume_ids: np.ndarray, top_k: int = 10) -> list[dict]:
    """
    Rank resumes by their similarity to a job description using resume embeddings, returning
    the top_k resume IDs and their similarity scores (e.g., [{"id": "R028", "score": 0.91}, ...]).
    """

    index = build_index(embeddings)  

    jd_vector = embed_text(jd_text) 
    jd_vector = jd_vector / np.linalg.norm(jd_vector) # normalize JD embedding for similarity comparison
    jd_vector = jd_vector.reshape(1, -1) # reshape JD vector for FAISS search

    scores, indices = index.search(jd_vector, top_k) # it gives the top k most similar resumes to JD embedding with scores (match %) and index of those resumes 

    results = []
    for score, idx in zip(scores[0], indices[0]): # loops through scores and the index of the topk resumes  .zip() pairs them  
        results.append({"id": resume_ids[idx], "score": float(score)})
    return results

    
def keyword_score(jd_terms: set, matched_terms: set) -> float:
    """
    Calculate the fraction of job-description terms found in the resume, given the required JD terms
    and resume text.
    """
    if not jd_terms:
        return 0.0
    return len(jd_terms & matched_terms) / len(jd_terms)


def missing_keywords(jd_terms: set, resume_text: str) -> list[str]:
    """
    Find and return the sorted list of JD terms that are missing from the resume.
    """
    found_terms = terms_present_in_text(jd_terms, resume_text)
    return sorted(jd_terms - found_terms)


def rank_resumes(jd_text: str, embeddings: np.ndarray, resume_ids: np.ndarray,
                  id_to_text: dict, semantic_weight: float = 0.4, top_k: int = 10) -> list[dict]:
    """
    Rank resumes by combining semantic similarity and keyword matching and return the
    top-ranked candidates with their scores and missing keywords.
    """
    # get semantic similarity scores for all resumes
    semantic_results = rank_by_similarity(jd_text, embeddings, resume_ids, top_k=len(resume_ids))
    jd_terms = extract_jd_terms(jd_text)
 
    final = []
    for r in semantic_results:
        # get skills belonging to the current resume,
        resume_text = id_to_text.get(r["id"], "")
        matched = terms_present_in_text(jd_terms, resume_text)
        # get the keyword score of match % of jd skills and resume skills 
        kw_score = keyword_score(jd_terms, matched)
        # combine both scores with resp. weight 
        combined = semantic_weight * r["score"] + (1 - semantic_weight) * kw_score
        final.append({
            "id": r["id"],
            "semantic_score": round(r["score"], 3),
            "keyword_score": round(kw_score, 3),
            "final_score": round(combined, 3),
            "missing_keywords": missing_keywords(jd_terms
                                                 , resume_text),
        })
    # Sort candidates using their final_score as the sorting key, with reverse=True
    # to arrange them from highest to lowest score.
    final.sort(key=lambda x: x["final_score"], reverse=True)
    return final[:top_k]


def rank_candidates(jd_text: str, embeddings: np.ndarray, resume_ids: np.ndarray,
                     resumes_df, semantic_weight: float = 0.4, top_k: int = 10) -> list[dict]:
    """
    Rank candidates from the resume DataFrame by converting stored skills into a mapping
    and using semantic similarity and keyword matching to return the top-ranked candidates
    with their scores and missing keywords.
    """
    # Skills are stored as strings in the CSV or the dataframe , so convert them back to lists.
    id_to_text = dict(zip(resumes_df["id"], resumes_df["raw_text"]))
    return rank_resumes(jd_text, embeddings, resume_ids, id_to_text, semantic_weight, top_k)
 
 
def rank_uploaded_resumes(jd_text: str, processed: dict, semantic_weight: float = 0.4,
                          top_k: int = 10) -> list[dict]:
    """
    Rank newly uploaded resumes using their processed skills and embeddings.
    """
    id_to_text = dict(zip(processed["ids"], processed["texts"])) # it zips the processed dict's ids and skills into one dict. 
    return rank_resumes(jd_text, processed["embeddings"], processed["ids"], id_to_text, semantic_weight, top_k)