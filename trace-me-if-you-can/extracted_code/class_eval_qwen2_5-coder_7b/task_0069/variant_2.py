import PyPDF2

class PDFMerger:
    def __init__(self, file_locations):
        self.file_locations = file_locations
        self.pdf_readers = [PyPDF2.PdfReader(location) for location in file_locations]

    def concatenate_pdfs(self, output_file_path):
        writer = PyPDF2.PdfWriter()

        for pdf_reader in self.pdf_readers:
            [writer.add_page(page) for page in pdf_reader.pages]

        with open(output_file_path, 'wb') as out:
            writer.write(out)
        return f"Merged PDFs saved to {output_file_path}"

    def extract_texts(self):
        all_texts = []
        for pdf_reader in self.pdf_readers:
            [all_texts.append(page.extract_text()) for page in pdf_reader.pages]
        return all_texts
