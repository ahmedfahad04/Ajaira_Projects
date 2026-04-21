import zipfile

class FileArchiver:
    def __init__(self, file_name):
        self.file_name = file_name

    def open_zip(self):
        try:
            return zipfile.ZipFile(self.file_name, 'r')
        except:
            return None

    def unzip_all(self, destination):
        try:
            with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
                zip_ref.extractall(destination)
            return True
        except:
            return False

    def unzip_specific(self, file_name, destination):
        try:
            with zipfile.ZipFile(self.file_name, 'r') as zip_ref:
                zip_ref.extract(file_name, destination)
            return True
        except:
            return False

    def make_zip(self, file_list, output_zip):
        try:
            with zipfile.ZipFile(output_zip, 'w') as zip_ref:
                for file in file_list:
                    zip_ref.write(file)
            return True
        except:
            return False
