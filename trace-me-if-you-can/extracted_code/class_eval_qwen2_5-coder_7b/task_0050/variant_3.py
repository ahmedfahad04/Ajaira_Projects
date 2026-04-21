import json
import os

class JSONFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_json(self):
        if not os.path.exists(self.file_path):
            return None
        try:
            with open(self.file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("Invalid JSON format")
            return None

    def write_json(self, data):
        try:
            with open(self.file_path, 'w') as file:
                json.dump(data, file, indent=4)
        except IOError as e:
            print(f"Error writing file: {e}")

    def modify_json(self, key_to_remove):
        data = self.read_json()
        if data is None:
            return False
        if key_to_remove in data:
            del data[key_to_remove]
            self.write_json(data)
            return True
        return False
