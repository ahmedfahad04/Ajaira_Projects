import csv
from contextlib import contextmanager


class CSVProcessor:

    @contextmanager
    def _open_csv_reader(self, file_name):
        with open(file_name, 'r') as file:
            yield csv.reader(file)

    @contextmanager  
    def _open_csv_writer(self, file_name):
        with open(file_name, 'w', newline='') as file:
            yield csv.writer(file)

    def read_csv(self, file_name):
        with self._open_csv_reader(file_name) as reader:
            title = next(reader)
            data = [row for row in reader]
        return title, data

    def write_csv(self, data, file_name):
        try:
            with self._open_csv_writer(file_name) as writer:
                writer.writerows(data)
            return 1
        except:
            return 0

    def process_csv_data(self, N, save_file_name):
        title, data = self.read_csv(save_file_name)
        column_data = [row[N] for row in data]
        column_data = [row.upper() for row in column_data]
        new_data = [title, column_data]
        return self.write_csv(new_data, save_file_name.split('.')[0] + '_process.csv')
