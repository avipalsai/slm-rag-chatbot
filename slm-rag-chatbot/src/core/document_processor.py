import PyPDF2
from docx import Document
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Parse and chunk documents for RAG ingestion"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return ""
    
    def parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
            return text
        except Exception as e:
            print(f"Error parsing DOCX {file_path}: {e}")
            return ""
    
    def parse_txt(self, file_path: str) -> str:
        """Extract text from TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error parsing TXT {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = self.splitter.split_text(text)
        return [c for c in chunks if len(c.strip()) > 50]
    
    def process_file(self, file_path: str) -> Dict:
        """
        Process any supported file type
        
        Returns:
            {
                'chunks': List[str],
                'metadata': {
                    'filename': str,
                    'file_type': str,
                    'total_chunks': int
                }
            }
        """
        if file_path.endswith('.pdf'):
            text = self.parse_pdf(file_path)
            file_type = 'pdf'
        elif file_path.endswith('.docx'):
            text = self.parse_docx(file_path)
            file_type = 'docx'
        elif file_path.endswith('.txt'):
            text = self.parse_txt(file_path)
            file_type = 'txt'
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        chunks = self.chunk_text(text)
        
        return {
            'chunks': chunks,
            'metadata': {
                'filename': file_path.split('\\')[-1],
                'file_type': file_type,
                'total_chunks': len(chunks)
            }
        }
