import PyPDF2


class PDFHandler:
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.readers = [PyPDF2.PdfReader(fp) for fp in filepaths]

    def merge_pdfs(self, output_filepath):
        pdf_writer = PyPDF2.PdfWriter()

        # Enumerate-based approach with explicit indexing
        for reader_idx, reader in enumerate(self.readers):
            page_count = len(reader.pages)
            for page_idx in range(page_count):
                current_page = reader.pages[page_idx]
                pdf_writer.add_page(current_page)

        with open(output_filepath, 'wb') as output_file:
            pdf_writer.write(output_file)
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        extracted_texts = []
        for reader_index in range(len(self.readers)):
            current_reader = self.readers[reader_index]
            for page_index in range(len(current_reader.pages)):
                page_content = current_reader.pages[page_index]
                extracted_texts.append(page_content.extract_text())
        return extracted_texts
