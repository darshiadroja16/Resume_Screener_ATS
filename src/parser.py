import pdfplumber       # read pdf 
from docx import Document  # read word files(.docx)
from pathlib import Path   # file extension read

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from all pages of a PDF file using its file path.
    """
    
    with pdfplumber.open(file_path) as pdf: # open the pdf

        return "\n".join(page.extract_text() or "" for page in pdf.pages) # read text from every page & convert it into 1 string 
    

def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from all para. of a DOCX file using its file path.
    """
    doc = Document(file_path)     # open the docx file 
    return "\n".join(p.text for p in doc.paragraphs)  # convert all para into 1 string 


def extract_text(file_path: str) -> str:
    """
    Detect the file type from its extension and extract text using the appropriate parser.
    """
    ext = Path(file_path).suffix.lower()  # get the file extension and convert to lower case 

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    if ext == ".docx":
        return extract_text_from_docx(file_path)

    raise ValueError(f"Unsupported file type: {ext}")


def extract_text_safe(file_path: str) -> str | None:
    """
    Safely extract text from a resume file and return None if extraction fails.
    """
    try:
        return extract_text(file_path)
    
    except Exception:
        return None
    