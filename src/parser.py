import pdfplumber       # read pdf 
from docx import Document  # read word files(.docx)
from pathlib import Path   # file extension read

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:   file_path (str): Path to the PDF file.
    Returns:  str: Extracted text from all pages of the PDF.
    """
    
    with pdfplumber.open(file_path) as pdf: # open the pdf

        return "\n".join(page.extract_text() or "" for page in pdf.pages) # read text from every page & convert it into 1 string 
    

def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a Word (.docx) file.

    Args:  file_path (str): Path to the DOCX file.
    Returns:   str: Extracted text from all paragraphs in the document.
    """
    doc = Document(file_path)     # open the docx file 
    return "\n".join(p.text for p in doc.paragraphs)  # convert all para into 1 string 


def extract_text(file_path: str) -> str:
    """
    Function automatically detects the file type
    and calls the appropriate parser to extract text.

    Args:   file_path (str): Path to the resume file.
    Returns:  str: Extracted text from the file.
    Raises:  ValueError: If the file format is not supported.
    """
    ext = Path(file_path).suffix.lower()  # get the file extension and convert to lower case 

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    if ext == ".docx":
        return extract_text_from_docx(file_path)

    raise ValueError(f"Unsupported file type: {ext}")


def extract_text_safe(file_path: str) -> str | None:
    """
    Safely extract text from a resume file, returning None if extraction fails.
    """
    try:
        return extract_text(file_path)
    
    except Exception:
        return None


# for testing
# if __name__ == "__main__":
    
#     file_path = r"C:\Users\HP\Desktop\Resume_Screener_ATS\Drashi Adroja AIML  Resume.pdf"

#     print(f"Testing file: {file_path}")
#     print("=" * 60)
 
#     text = extract_text_safe(file_path)

#     if text is None:
#         print("Failed to extract text.")
#     else:
#         print("Text extracted successfully!\n")
#         print(text)
#         print(f"Total Characters Extracted: {len(text)}")