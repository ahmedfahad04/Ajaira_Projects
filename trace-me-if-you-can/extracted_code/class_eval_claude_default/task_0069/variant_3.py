import PyPDF2
from functools import reduce
from operator import add


class PDFHandler:
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.readers = [PyPDF2.PdfReader(fp) for fp in filepaths]

    def merge_pdfs(self, output_filepath):
        pdf_writer = PyPDF2.PdfWriter()

        # Functional approach using map and reduce
        page_lists = map(lambda reader: list(reader.pages), self.readers)
        all_pages = reduce(add, page_lists, [])
        
        list(map(pdf_writer.add_page, all_pages))

        with open(output_filepath, 'wb') as out:
            pdf_writer.write(out)
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        # Functional composition approach
        extract_from_reader = lambda reader: [page.extract_text() for page in reader.pages]
        text_lists = map(extract_from_reader, self.readers)
        return reduce(add, text_lists, [])
