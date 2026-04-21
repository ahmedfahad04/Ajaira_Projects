import json
import os


class JSONProcessor:
    SUCCESS = 1
    FILE_NOT_FOUND = 0
    ERROR = -1

    def read_json(self, file_path):
        return (
            self.FILE_NOT_FOUND if not os.path.exists(file_path)
            else self._attempt_json_read(file_path)
        )

    def _attempt_json_read(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except:
            return self.ERROR

    def write_json(self, data, file_path):
        return self._attempt_json_write(data, file_path)

    def _attempt_json_write(self, data, file_path):
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return self.SUCCESS
        except:
            return self.ERROR

    def process_json(self, file_path, remove_key):
        data = self.read_json(file_path)
        return (
            self.FILE_NOT_FOUND if data in [self.FILE_NOT_FOUND, self.ERROR]
            else self._process_key_removal(data, remove_key, file_path)
        )

    def _process_key_removal(self, data, remove_key, file_path):
        if remove_key not in data:
            return self.FILE_NOT_FOUND
        del data[remove_key]
        self.write_json(data, file_path)
        return self.SUCCESS
