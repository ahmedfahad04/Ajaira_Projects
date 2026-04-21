import json
import os

class JsonFile:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if not os.path.exists(self.file_path):
            return None
        try:
            with open(self.file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("Invalid JSON data")
            return None

    def save(self, data):
        try:
            with open(self.file_path, 'w') as file:
                json.dump(data, file, indent=4)
        except IOError as e:
            print(f"Error writing file: {e}")

    def modify(self, key_to_remove):
        data = self.load()
        if data is None:
            return False
        if key_to_remove in data:
            del data[key_to_remove]
            self.save(data)
            return True
        return False
