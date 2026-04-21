import json
import os


class JSONProcessor:
    def _handle_file_operation(self, operation, *args):
        try:
            return operation(*args)
        except:
            return -1

    def read_json(self, file_path):
        if not os.path.exists(file_path):
            return 0
        
        def _read_operation():
            with open(file_path, 'r') as file:
                return json.load(file)
        
        return self._handle_file_operation(_read_operation)

    def write_json(self, data, file_path):
        def _write_operation():
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return 1
        
        return self._handle_file_operation(_write_operation)

    def process_json(self, file_path, remove_key):
        data = self.read_json(file_path)
        if data in [0, -1]:
            return 0
        
        modified = data.pop(remove_key, None) is not None
        if modified:
            self.write_json(data, file_path)
            return 1
        return 0
