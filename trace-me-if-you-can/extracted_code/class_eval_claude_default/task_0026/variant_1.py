import csv
from pathlib import Path


class CSVProcessor:
    
    def read_csv(self, file_name):
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
        return rows[0], rows[1:]

    def write_csv(self, data, file_name):
        try:
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)
            return 1
        except:
            return 0

    def process_csv_data(self, N, save_file_name):
        title, data = self.read_csv(save_file_name)
        processed_column = [row[N].upper() for row in data]
        output_data = [title, processed_column]
        output_path = Path(save_file_name).stem + '_process.csv'
        return self.write_csv(output_data, output_path)
