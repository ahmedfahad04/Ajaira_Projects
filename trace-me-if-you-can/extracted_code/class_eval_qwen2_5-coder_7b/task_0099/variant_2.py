import zipfile
from pathlib import Path

class ZipHandler:
    def __init__(self, zip_file):
        self.zip_file = Path(zip_file)

    def load_zip(self):
        try:
            return zipfile.ZipFile(self.zip_file, 'r')
        except:
            return None

    def unzip_all(self, extraction_path):
        extraction_path = Path(extraction_path)
        try:
            with zipfile.ZipFile(self.zip_file, 'r') as archive:
                archive.extractall(extraction_path)
            return True
        except:
            return False

    def unzip_file(self, file_name, output_folder):
        output_folder = Path(output_folder)
        try:
            with zipfile.ZipFile(self.zip_file, 'r') as archive:
                archive.extract(file_name, output_folder)
            return True
        except:
            return False

    def generate_zip(self, files, new_zip):
        new_zip_path = Path(new_zip)
        try:
            with zipfile.ZipFile(new_zip_path, 'w') as archive:
                for file in files:
                    archive.write(file)
            return True
        except:
            return False
