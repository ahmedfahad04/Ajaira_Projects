import csv

class CSVManager:
    def extract_data(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            titles = next(reader)
            rows = [row for row in reader]
        return titles, rows

    def save_modified_data(self, file_path, data):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        return 1 if not file.closed else 0

    def update_csv_data(self, column, input_file, output_file):
        titles, rows = self.extract_data(input_file)
        updated_column = [item[column].upper() for item in rows]
        new_data = [titles, updated_column]
        return self.save_modified_data(output_file, new_data)
