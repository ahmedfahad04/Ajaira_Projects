import PyPDF2

class DocumentMerger:
    def __init__(self, document_paths):
        self.document_paths = document_paths
        self.document_readers = [PyPDF2.PdfReader(path) for path in document_paths]

    def combine_documents(self, output_location):
        combined_writer = PyPDF2.PdfWriter()

        for doc_reader in self.document_readers:
            for page_index in range(len(doc_reader.pages)):
                page = doc_reader.pages[page_index]
                combined_writer.add_page(page)

        with open(output_location, 'wb') as output_file:
            combined_writer.write(output_file)
        return f"Documents merged at {output_location}"

    def gather_texts(self):
        collected_texts = []
        for doc_reader in self.document_readers:
            for page_index in range(len(doc_reader.pages)):
                page = doc_reader.pages[page_index]
                collected_texts.append(page.extract_text())
        return collected_texts
