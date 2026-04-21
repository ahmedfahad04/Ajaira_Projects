import PyPDF2
from itertools import chain


class PDFHandler:
    def __init__(self, filepaths):
        self.filepaths = filepaths

    @property
    def readers(self):
        """Property-based lazy loading of PDF readers"""
        if not hasattr(self, '_readers'):
            self._readers = [PyPDF2.PdfReader(fp) for fp in self.filepaths]
        return self._readers

    def merge_pdfs(self, output_filepath):
        pdf_writer = PyPDF2.PdfWriter()

        # Use itertools.chain for flattening pages
        pages_iterator = chain.from_iterable(reader.pages for reader in self.readers)
        
        # Process pages in batches (though batch size is effectively unlimited here)
        for page in pages_iterator:
            pdf_writer.add_page(page)

        with open(output_filepath, 'wb') as out:
            pdf_writer.write(out)
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        # Iterator-based approach with chain
        pages_chain = chain.from_iterable(reader.pages for reader in self.readers)
        return [page.extract_text() for page in pages_chain]
