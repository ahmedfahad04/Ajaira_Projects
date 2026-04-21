import csv

class DataHandler:
    def read_data_from_csv(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            data = [row for row in reader]
        return header, data

    def write_data_to_csv(self, data, output_path):
        try:
            with open(output_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)
            return 1
        except Exception as e:
            return 0

    def process_data(self, column_index, input_file, output_file):
        header, data = self.read_data_from_csv(input_file)
        column_data = [row[column_index].upper() for row in data]
        new_data = [header, column_data]
        return self.write_data_to_csv(new_data, output_file)
