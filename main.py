import shutil
import tempfile
from pathlib import Path 

from fastapi import FastAPI , UploadFile , Form 
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import process_uploaded_resumes
from src.ranker import rank_uploaded_resumes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    #allows react frontend to access the backend or make request to backend 
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"], # all methods are allowed 
    allow_headers=["*"] # headers are the extra info about server or what type of data is being sent , all this headers are allowed 
)


@app.get("/")
def home():
    return {"message": "Resume Screener API is running "}


@app.post("/rank")
async def rank(jd_text: str = Form(...), files: list[UploadFile] = None):
    """
    Receive a job description and uploaded resumes, process and 
    rank the resumes, and return the ranked results as JSON.
    """
    if not jd_text.strip():
        return {"error": "JD text is required"}
    if not files:
        return {"error": "At least one resume file is required"}
    
    # create a temporary folder for the uploaded resumes.
    with tempfile.TemporaryDirectory() as tmp_dir:

        # save each uploaded resume into the temporary folder
        for file in files:
            dest = Path(tmp_dir) / file.filename

            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)

        # process the uploaded resumes using the existing pipeline.
        processed = process_uploaded_resumes(tmp_dir)

        # rank the processed resumes against the job description.
        results = rank_uploaded_resumes(
            jd_text, 
            processed,
            top_k=len(processed["ids"])
        )

    # return the ranked results as JSON.
    return {"results": results}