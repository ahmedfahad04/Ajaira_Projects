import json

class FileProcessor:
    def __init__(self, path):
        self.path = path

    def load_json_data(self):
        with open(self.path, 'r') as file:
            return json.load(file)

    def read_file(self):
        with open(self.path, 'r') as file:
            return file.read()

    def save_file(self, text):
        with open(self.path, 'w') as file:
            file.write(text)

    def clean_and_save(self):
        raw_content = self.read_file()
        cleaned_content = ''.join(filter(str.isalpha, raw_content))
        self.save_file(cleaned_content)
        return cleaned_content
