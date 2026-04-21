import csv

class DataCSV:
    def fetch_csv_data(self, file_name):
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            titles = next(reader)
            data = [row for row in reader]
        return titles, data

    def output_csv_data(self, data, file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        return 1 if not file.closed else 0

    def handle_csv(self, column_index, input_file, output_file):
        titles, data = self.fetch_csv_data(input_file)
        updated_data = [[row[column_index].upper() for row in data]]
        new_data = [titles, updated_data[0]]
        return self.output_csv_data(new_data, output_file)
