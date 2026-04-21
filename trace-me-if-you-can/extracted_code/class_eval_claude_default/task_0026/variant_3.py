import csv


class CSVProcessor:

    def __init__(self):
        self._cache = {}

    def read_csv(self, file_name):
        if file_name in self._cache:
            return self._cache[file_name]
        
        data = []
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            title = next(reader)
            data.extend(reader)
        
        result = (title, data)
        self._cache[file_name] = result
        return result

    def write_csv(self, data, file_name):
        try:
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)
            return 1
        except Exception:
            return 0

    def process_csv_data(self, N, save_file_name):
        title, data = self.read_csv(save_file_name)
        transformed_column = list(map(lambda row: row[N].upper(), data))
        result_data = [title, transformed_column]
        base_name = save_file_name.rsplit('.', 1)[0]
        return self.write_csv(result_data, f"{base_name}_process.csv")
