import csv

class CSVModifier:
    def read_csv_rows(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            rows = list(reader)
        return headers, rows

    def write_csv_rows(self, file_path, data):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        return 1 if not file.closed else 0

    def transform_and_save_csv(self, column_index, input_csv, output_csv):
        headers, rows = self.read_csv_rows(input_csv)
        updated_rows = [[row[column_index].upper()] for row in rows]
        new_data = [headers, updated_rows]
        return self.write_csv_rows(output_csv, new_data)
