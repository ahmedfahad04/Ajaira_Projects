import PyPDF2

class PDFCombiner:
    def __init__(self, file_list):
        self.file_list = file_list
        self.readers = [PyPDF2.PdfReader(file) for file in file_list]

    def merge_and_save(self, output_file):
        writer = PyPDF2.PdfWriter()

        for reader in self.readers:
            for page in reader.pages:
                writer.add_page(page)

        with open(output_file, 'wb') as output:
            writer.write(output)
        return f"Merged PDFs saved at {output_file}"

    def extract_from_pdfs(self):
        texts = []
        for reader in self.readers:
            for page in reader.pages:
                texts.append(page.extract_text())
        return texts
