import PyPDF2
from typing import List


class PDFHandler:
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self._load_readers()

    def _load_readers(self):
        """Lazy loading helper method"""
        self.readers = [PyPDF2.PdfReader(fp) for fp in self.filepaths]

    def _get_all_pages(self):
        """Extract all pages from all PDF readers"""
        pages = []
        for reader in self.readers:
            pages.extend(reader.pages)
        return pages

    def merge_pdfs(self, output_filepath):
        pdf_writer = PyPDF2.PdfWriter()
        
        all_pages = self._get_all_pages()
        for page in all_pages:
            pdf_writer.add_page(page)

        with open(output_filepath, 'wb') as out:
            pdf_writer.write(out)
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        all_pages = self._get_all_pages()
        return [page.extract_text() for page in all_pages]
