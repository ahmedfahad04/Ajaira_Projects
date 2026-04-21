import json


def create_text_file_processor(file_path):
    def read_file_as_json():
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def read_file():
        with open(file_path, 'r') as file:
            return file.read()

    def write_file(content):
        with open(file_path, 'w') as file:
            file.write(content)

    def process_file():
        content = read_file()
        content = ''.join([char for char in content if char.isalpha()])
        write_file(content)
        return content

    processor = type('TextFileProcessor', (), {})()
    processor.read_file_as_json = read_file_as_json
    processor.read_file = read_file
    processor.write_file = write_file
    processor.process_file = process_file
    return processor


class TextFileProcessor:
    def __init__(self, file_path):
        self._processor = create_text_file_processor(file_path)

    def read_file_as_json(self):
        return self._processor.read_file_as_json()

    def read_file(self):
        return self._processor.read_file()

    def write_file(self, content):
        return self._processor.write_file(content)

    def process_file(self):
        return self._processor.process_file()
