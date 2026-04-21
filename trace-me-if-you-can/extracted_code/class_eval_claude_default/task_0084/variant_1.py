import json
from pathlib import Path


class TextFileProcessor:
    def __init__(self, file_path):
        self._path = Path(file_path)

    def read_file_as_json(self):
        return json.loads(self._path.read_text())

    def read_file(self):
        return self._path.read_text()

    def write_file(self, content):
        self._path.write_text(content)

    def process_file(self):
        content = self.read_file()
        processed_content = ''.join(filter(str.isalpha, content))
        self.write_file(processed_content)
        return processed_content
