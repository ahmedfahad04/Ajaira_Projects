import csv


def read_csv_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        title = next(reader)
        for row in reader:
            data.append(row)
    return title, data


def write_csv_file(data, file_name):
    try:
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        return 1
    except:
        return 0


class CSVProcessor:

    def read_csv(self, file_name):
        return read_csv_file(file_name)

    def write_csv(self, data, file_name):
        return write_csv_file(data, file_name)

    def process_csv_data(self, N, save_file_name):
        title, data = self.read_csv(save_file_name)
        column_data = [row[N] for row in data]
        column_data = [row.upper() for row in column_data]
        new_data = [title, column_data]
        return self.write_csv(new_data, save_file_name.split('.')[0] + '_process.csv')
