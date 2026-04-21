import json

class DocumentHandler:
    def __init__(self, doc_path):
        self.doc_path = doc_path

    def get_json_data(self):
        with open(self.doc_path, 'r') as file:
            return json.load(file)

    def get_file_content(self):
        with open(self.doc_path, 'r') as file:
            return file.read()

    def update_file(self, new_content):
        with open(self.doc_path, 'w') as file:
            file.write(new_content)

    def refine_and_update(self):
        full_content = self.get_file_content()
        refined_content = ''.join(filter(str.isalpha, full_content))
        self.update_file(refined_content)
        return refined_content
