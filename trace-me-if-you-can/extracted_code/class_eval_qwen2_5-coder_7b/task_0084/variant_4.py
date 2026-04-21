import json

class FileModifier:
    def __init__(self, file_loc):
        self.file_loc = file_loc

    def parse_json(self):
        with open(self.file_loc, 'r') as file:
            return json.load(file)

    def fetch_file_data(self):
        with open(self.file_loc, 'r') as file:
            return file.read()

    def save_data(self, text):
        with open(self.file_loc, 'w') as file:
            file.write(text)

    def clean_and_save_data(self):
        file_data = self.fetch_file_data()
        cleaned_data = ''.join(filter(str.isalpha, file_data))
        self.save_data(cleaned_data)
        return cleaned_data
