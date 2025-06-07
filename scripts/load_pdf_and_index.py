from services.pdf_ingestor import extract_text_from_pdf
from services.rag_engine import create_vectorstore_from_text
 
if __name__ == "__main__":
    path_to_pdf = "data/sample_nbfc_report.pdf"
    text = extract_text_from_pdf(path_to_pdf)
    vectorstore = create_vectorstore_from_text(text)
    print("FAISS index created and saved.") 