import csv


class CSVFileHandler:
    
    @staticmethod
    def read(file_name):
        data = []
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            title = next(reader)
            for row in reader:
                data.append(row)
        return title, data
    
    @staticmethod
    def write(data, file_name):
        try:
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)
            return 1
        except:
            return 0


class CSVProcessor:

    def __init__(self):
        self.file_handler = CSVFileHandler()

    def read_csv(self, file_name):
        return self.file_handler.read(file_name)

    def write_csv(self, data, file_name):
        return self.file_handler.write(data, file_name)

    def process_csv_data(self, N, save_file_name):
        title, data = self.read_csv(save_file_name)
        column_data = [row[N] for row in data]
        column_data = [row.upper() for row in column_data]
        new_data = [title, column_data]
        return self.write_csv(new_data, save_file_name.split('.')[0] + '_process.csv')
