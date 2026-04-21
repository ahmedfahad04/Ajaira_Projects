import PyPDF2

class PDFManager:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.pdf_readers = [PyPDF2.PdfReader(path) for path in file_paths]

    def merge_pdfs_to(self, output_path):
        merged_writer = PyPDF2.PdfWriter()

        for reader in self.pdf_readers:
            for page in reader.pages:
                merged_writer.add_page(page)

        with open(output_path, 'wb') as output_file:
            merged_writer.write(output_file)
        return f"Merged PDFs saved at {output_path}"

    def extract_texts_from_pdfs(self):
        extracted_texts = []
        for reader in self.pdf_readers:
            for page in reader.pages:
                extracted_texts.append(page.extract_text())
        return extracted_texts
