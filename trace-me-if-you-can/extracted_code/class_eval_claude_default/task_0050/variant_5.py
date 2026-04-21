import json
import os


class JSONProcessor:
    def __init__(self):
        self.file_operations = {
            'read': self._read_file,
            'write': self._write_file
        }

    def _read_file(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def _write_file(self, file_path, data):
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def _execute_operation(self, operation, *args):
        try:
            return self.file_operations[operation](*args)
        except:
            return None

    def read_json(self, file_path):
        if not os.path.exists(file_path):
            return 0
        result = self._execute_operation('read', file_path)
        return result if result is not None else -1

    def write_json(self, data, file_path):
        result = self._execute_operation('write', file_path, data)
        return 1 if result is not None else -1

    def process_json(self, file_path, remove_key):
        data = self.read_json(file_path)
        if data == 0 or data == -1:
            return 0
        
        key_exists = remove_key in data
        if key_exists:
            data.pop(remove_key)
            self.write_json(data, file_path)
        
        return 1 if key_exists else 0
