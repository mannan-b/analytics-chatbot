import logging
from typing import Dict, Any, List
from PyPDF2 import PdfReader
from io import BytesIO

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.chunk_size = 300
    
    async def process_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        try:
            pdf_file = BytesIO(pdf_bytes)
            reader = PdfReader(pdf_file)
            
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            
            full_text = self._clean_text(full_text)
            chunks = self._chunk_text(full_text, self.chunk_size)
            
            logger.info(f"Processed {filename}: {len(reader.pages)} pages → {len(chunks)} chunks")
            
            return {
                'success': True,
                'filename': filename,
                'total_chars': len(full_text),
                'total_pages': len(reader.pages),
                'chunks': chunks
            }
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _clean_text(self, text: str) -> str:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
