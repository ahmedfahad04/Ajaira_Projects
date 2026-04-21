import json
import os

class DataHandler:
    def load_data(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def save_data(self, data, path):
        try:
            with open(path, 'w') as file:
                json.dump(data, file)
        except Exception as e:
            print(f"Error saving data: {e}")

    def modify_data(self, path, key_to_remove):
        data = self.load_data(path)
        if data is None:
            return False
        if key_to_remove in data:
            del data[key_to_remove]
            self.save_data(data, path)
            return True
        return False
