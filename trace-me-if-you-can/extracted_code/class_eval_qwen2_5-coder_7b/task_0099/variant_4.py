import zipfile
from pathlib import Path

class ZipUtility:
    def __init__(self, zip_file):
        self.zip_file = Path(zip_file)

    def open_zip(self):
        try:
            return zipfile.ZipFile(self.zip_file, 'r')
        except:
            return None

    def unzip_all(self, output_dir):
        output_dir = Path(output_dir)
        try:
            with zipfile.ZipFile(self.zip_file, 'r') as archive:
                archive.extractall(output_dir)
            return True
        except:
            return False

    def unzip_file(self, file_name, output_dir):
        output_dir = Path(output_dir)
        try:
            with zipfile.ZipFile(self.zip_file, 'r') as archive:
                archive.extract(file_name, output_dir)
            return True
        except:
            return False

    def create_zip(self, files, output_zip):
        output_zip_path = Path(output_zip)
        try:
            with zipfile.ZipFile(output_zip_path, 'w') as archive:
                for file in files:
                    archive.write(file)
            return True
        except:
            return False
