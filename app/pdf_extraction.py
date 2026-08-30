import pdfplumber
 
 
def extract_pages(pdf_path):
    """Extract text per page from a PDF as a list of (page_number, text) pairs.
 
    Raises on read/parse failure — the caller decides whether to skip the
    file and continue, since that's an orchestration decision, not this
    function's job.
    """
    with pdfplumber.open(pdf_path) as pdf:
        return [(i, page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)]