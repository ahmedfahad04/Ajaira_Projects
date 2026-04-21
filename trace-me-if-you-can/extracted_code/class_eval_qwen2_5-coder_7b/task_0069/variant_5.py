import PyPDF2

class PDFCombiner:
    def __init__(self, document_files):
        self.document_files = document_files
        self.document_readers = {file: PyPDF2.PdfReader(file) for file in document_files}

    def combine_documents_and_save(self, output_location):
        combined_writer = PyPDF2.PdfWriter()

        for reader in self.document_readers.values():
            for page in reader.pages:
                combined_writer.add_page(page)

        with open(output_location, 'wb') as output_file:
            combined_writer.write(output_file)
        return f"Documents merged at {output_location}"

    def extract_texts_from_pdfs(self):
        all_texts = []
        for reader in self.document_readers.values():
            for page in reader.pages:
                all_texts.append(page.extract_text())
        return all_texts
