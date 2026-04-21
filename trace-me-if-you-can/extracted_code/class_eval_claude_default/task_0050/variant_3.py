import json
import os
from contextlib import contextmanager


class JSONProcessor:
    @contextmanager
    def _safe_file_operation(self, file_path, mode):
        try:
            with open(file_path, mode) as file:
                yield file
        except:
            yield None

    def read_json(self, file_path):
        if not os.path.exists(file_path):
            return 0
        
        with self._safe_file_operation(file_path, 'r') as file:
            if file is None:
                return -1
            try:
                return json.load(file)
            except:
                return -1

    def write_json(self, data, file_path):
        with self._safe_file_operation(file_path, 'w') as file:
            if file is None:
                return -1
            try:
                json.dump(data, file)
                return 1
            except:
                return -1

    def process_json(self, file_path, remove_key):
        data = self.read_json(file_path)
        if data == 0 or data == -1:
            return 0
        if remove_key in data:
            del data[remove_key]
            self.write_json(data, file_path)
            return 1
        else:
            return 0
