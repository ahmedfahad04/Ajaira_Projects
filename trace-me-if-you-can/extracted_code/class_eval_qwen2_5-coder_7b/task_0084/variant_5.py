import json

class ContentHandler:
    def __init__(self, filepath):
        self.filepath = filepath

    def get_json(self):
        with open(self.filepath, 'r') as file:
            return json.load(file)

    def get_content(self):
        with open(self.filepath, 'r') as file:
            return file.read()

    def update_content(self, content):
        with open(self.filepath, 'w') as file:
            file.write(content)

    def refine_and_update_content(self):
        content = self.get_content()
        refined_content = ''.join([char for char in content if char.isalpha()])
        self.update_content(refined_content)
        return refined_content
