import PyPDF2
from pathlib import Path


class PDFHandler:
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.readers = [PyPDF2.PdfReader(fp) for fp in filepaths]

    def merge_pdfs(self, output_filepath):
        pdf_writer = PyPDF2.PdfWriter()
        
        # Flatten all pages from all readers into a single iterator
        all_pages = (page for reader in self.readers for page in reader.pages)
        
        for page in all_pages:
            pdf_writer.add_page(page)

        Path(output_filepath).write_bytes(pdf_writer.getvalue())
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        # Generator expression for memory efficiency
        return [page.extract_text() 
                for reader in self.readers 
                for page in reader.pages]
