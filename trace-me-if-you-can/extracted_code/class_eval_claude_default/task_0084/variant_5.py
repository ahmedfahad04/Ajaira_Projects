import json
from io import StringIO


class TextFileProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self._file_operations = {
            'read': self._read_operation,
            'write': self._write_operation,
            'json_read': self._json_read_operation
        }

    def _read_operation(self, file_handle):
        return file_handle.read()

    def _write_operation(self, file_handle, content=None):
        if content is not None:
            file_handle.write(content)

    def _json_read_operation(self, file_handle):
        return json.load(file_handle)

    def _execute_file_operation(self, mode, operation_key, content=None):
        with open(self.file_path, mode) as file:
            operation = self._file_operations[operation_key]
            if content is not None:
                return operation(file, content)
            return operation(file)

    def read_file_as_json(self):
        return self._execute_file_operation('r', 'json_read')

    def read_file(self):
        return self._execute_file_operation('r', 'read')

    def write_file(self, content):
        self._execute_file_operation('w', 'write', content)

    def process_file(self):
        content = self.read_file()
        buffer = StringIO()
        for char in content:
            if char.isalpha():
                buffer.write(char)
        processed_content = buffer.getvalue()
        buffer.close()
        self.write_file(processed_content)
        return processed_content
