import json
import os
from pathlib import Path


class JSONProcessor:
    def read_json(self, file_path):
        path = Path(file_path)
        if not path.exists():
            return 0
        try:
            return json.loads(path.read_text())
        except:
            return -1

    def write_json(self, data, file_path):
        try:
            Path(file_path).write_text(json.dumps(data))
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
