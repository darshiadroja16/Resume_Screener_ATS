import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> np.ndarray:
    """
    Convert text into a numerical embedding.

    Args: text (str): Text to convert into an embedding.
    Returns: np.ndarray: Embedding vector representing the text.
    """
    return _model.encode([text])[0] # coverts the text into embedding vector



def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index from resume embeddings.

    Args: np.ndarray: Array containing resume embeddings.
    Returns: faiss.IndexFlatIP: FAISS index containing the resume embeddings.
    """
    # normalization makes embedding's length/norm  1 (length/norm is the square root of the sum of squares of all elements in the vector)
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # embedding is diveded by norm to make it normalized  
    index = faiss.IndexFlatIP(normalized.shape[1]) # creates the faiss index using inner-product(IP) similarity search method.shape[1] gives dimension of embedding
    index.add(normalized) # adds embedding vectors to index / faiss
    return index


def rank_by_similarity(jd_text: str, embeddings: np.ndarray, resume_ids: np.ndarray, top_k: int = 10):
    """
    Rank resumes based on similarity to a job description.

    Args:
        jd_text (str): Job description text.
        embeddings (np.ndarray): Resume embeddings.
        resume_ids (np.ndarray): IDs corresponding to the embeddings.
        top_k (int): Number of top resumes to return.
    Returns:
        list[dict]: Ranked resume IDs and their similarity scores.
        [{"id": "R028", "score": 0.91},....]
    """

    index = build_index(embeddings) # builds the faiss index using resume embeddings 

    jd_vector = embed_text(jd_text) # embed the jd text
    jd_vector = jd_vector / np.linalg.norm(jd_vector) # normalize JD embedding for similarity comparison
    jd_vector = jd_vector.reshape(1, -1) # reshape JD vector for FAISS search

    scores, indices = index.search(jd_vector, top_k) # it gives the topk most similar resumes to JD embedding with scores (match %) and index of those resumes 

    results = []
    for score, idx in zip(scores[0], indices[0]): # loops through scores and the index of the topk resumes.zip() pairs them  
        results.append({"id": resume_ids[idx], "score": float(score)})
    return results

    
    # testing code 
    # uv run python -c "import numpy as np; from src.ranker import rank_by_similarity; embeddings = np.load('models/resume_embeddings.npy'); resume_ids = np.load('models/resume_ids.npy', allow_pickle=True); jd = 'Looking for a Python developer with machine learning and NLP experience'; results = rank_by_similarity(jd, embeddings, resume_ids, top_k=5); [print(r) for r in results]"

def keyword_score(jd_skills: set, resume_skills: set) -> float:
    """
    Calculate the percentage of JD skills found in the resume.

    Args:
        jd_skills (set): Skills required by the job description.
        resume_skills (set): Skills found in the resume.
    Returns: float: Fraction of JD skills present in the resume.
    """
    if not jd_skills:
        return 0.0
    return len(jd_skills & resume_skills) / len(jd_skills)
 
 
def missing_keywords(jd_skills: set, resume_skills: set) -> list[str]:
    """
    Find skills required by the JD that are missing from the resume.

    Args:
        jd_skills (set): Skills required by the job description.
        resume_skills (set): Skills found in the resume.
    Returns: 
        list[str]: Sorted list of missing skills.
    """
    return sorted(jd_skills - resume_skills) # returns the skills that are in jd skills but not in resume skills 
 
 
def rank_candidates(jd_text: str, embeddings: np.ndarray, resume_ids: np.ndarray,
                     resumes_df, semantic_weight: float = 0.4, top_k: int = 10) -> list[dict] :
    """
    Rank candidates using semantic similarity and keyword matching.

    Args:
        jd_text (str): Job description text.
        embeddings (np.ndarray): Resume embeddings.
        resume_ids (np.ndarray): Resume IDs.
        resumes_df: DataFrame containing resume IDs and extracted skills(parsed_resumes.csv).
        semantic_weight (float): Weight given to semantic similarity.
        top_k (int): Number of top candidates to return.
    Returns:
        list[dict]: Ranked candidates with scores and missing keywords.
    """
    import ast
    from src.preprocess import extract_skills

    # get semantic similarity scores for all resumes
    semantic_results = rank_by_similarity(jd_text, embeddings, resume_ids, top_k=len(resume_ids))
    jd_skills = set(extract_skills(jd_text))

    #  Map each resume ID to its stored skills
    id_to_skills = dict(zip(resumes_df["id"], resumes_df["skills"]))
 
    final = []

    # calculate scores for each resume
    for r in semantic_results:

        # convert stored skills string into a set
        resume_skills = set(ast.literal_eval(id_to_skills.get(r["id"], "[]")))
        # calculate keyword matching score
        kw_score = keyword_score(jd_skills, resume_skills)
        combined = semantic_weight * r["score"] + (1 - semantic_weight) * kw_score
        final.append({
            "id": r["id"],
            "semantic_score": round(r["score"], 3),
            "keyword_score": round(kw_score, 3),
            "final_score": round(combined, 3),
            "missing_keywords": missing_keywords(jd_skills, resume_skills),
        })
 
    final.sort(key=lambda x: x["final_score"], reverse=True)
    return final[:top_k]