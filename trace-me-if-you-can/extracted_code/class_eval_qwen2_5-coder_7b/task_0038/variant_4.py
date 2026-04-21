import openpyxl


class ExcelReader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = None

    def load_data(self):
        try:
            workbook = openpyxl.load_workbook(self.file_name)
            self.data = workbook.active.values_only
            workbook.close()
        except Exception as e:
            print(f"Failed to load data: {e}")
            self.data = None


class ExcelWriter:
    def __init__(self, file_name):
        self.file_name = file_name

    def save_data(self, data):
        try:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            for row in data:
                sheet.append(row)
            workbook.save(self.file_name)
            workbook.close()
            return True
        except Exception as e:
            print(f"Failed to save data: {e}")
            return False


class ExcelProcessor:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    def process_data(self, N):
        new_data = [
            [*row, str(row[N]).upper() if not str(row[N]).isdigit() else row[N]]
            for row in self.reader.data
        ]
        return new_data

    def process_excel_data(self, N):
        self.reader.load_data()
        if self.reader.data is None or N >= len(next(self.reader.data)):
            return 0
        new_data = self.process_data(N)
        new_file_name = self.reader.file_name.split('.')[0] + '_processed.xlsx'
        return self.writer.save_data(new_data), new_file_name
