import json
import os

class JsonModifier:
    def read_json_file(self, file_path):
        if not os.path.isfile(file_path):
            return None
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("Invalid JSON format")
            return None

    def write_json_file(self, data, file_path):
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
        except IOError as e:
            print(f"Error writing file: {e}")

    def process_json(self, file_path, key_to_remove):
        data = self.read_json_file(file_path)
        if data is None:
            return False
        if key_to_remove in data:
            del data[key_to_remove]
            self.write_json_file(data, file_path)
            return True
        return False
