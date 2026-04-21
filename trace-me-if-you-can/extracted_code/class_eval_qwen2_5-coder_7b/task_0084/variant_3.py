import json

class DataModifier:
    def __init__(self, file_location):
        self.file_location = file_location

    def extract_json_data(self):
        with open(self.file_location, 'r') as file:
            return json.load(file)

    def load_file(self):
        with open(self.file_location, 'r') as file:
            return file.read()

    def write_file(self, text):
        with open(self.file_location, 'w') as file:
            file.write(text)

    def modify_and_store(self):
        file_content = self.load_file()
        only_letters = ''.join([char for char in file_content if char.isalpha()])
        self.write_file(only_letters)
        return only_letters
