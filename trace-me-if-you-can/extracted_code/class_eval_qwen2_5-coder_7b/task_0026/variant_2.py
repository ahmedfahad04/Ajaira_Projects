import csv

class CSVUtility:
    def load_csv(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            records = list(reader)
        return headers, records

    def save_csv(self, filename, content):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(content)
        return 1

    def transform_and_export_csv(self, index, input_csv, output_csv):
        headers, records = self.load_csv(input_csv)
        transformed_data = [[headers[index].upper()] + record[index:index+1] for record in records]
        return self.save_csv(output_csv, transformed_data)
