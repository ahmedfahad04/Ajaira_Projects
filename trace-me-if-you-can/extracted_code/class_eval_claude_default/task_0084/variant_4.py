import json
from contextlib import contextmanager


class TextFileProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    @contextmanager
    def _file_reader(self):
        file = open(self.file_path, 'r')
        try:
            yield file
        finally:
            file.close()

    @contextmanager
    def _file_writer(self):
        file = open(self.file_path, 'w')
        try:
            yield file
        finally:
            file.close()

    def read_file_as_json(self):
        with self._file_reader() as file:
            data = json.load(file)
        return data

    def read_file(self):
        with self._file_reader() as file:
            return file.read()

    def write_file(self, content):
        with self._file_writer() as file:
            file.write(content)

    def process_file(self):
        content = self.read_file()
        alphabetic_chars = []
        for char in content:
            if char.isalpha():
                alphabetic_chars.append(char)
        content = ''.join(alphabetic_chars)
        self.write_file(content)
        return content
