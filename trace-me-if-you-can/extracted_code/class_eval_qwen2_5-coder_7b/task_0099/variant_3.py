import zipfile
from pathlib import Path

class FileZipper:
    def __init__(self, zip_path):
        self.zip_path = Path(zip_path)

    def open_zip(self):
        try:
            return zipfile.ZipFile(self.zip_path, 'r')
        except:
            return None

    def extract_zip(self, output_dir):
        output_dir = Path(output_dir)
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as archive:
                archive.extractall(output_dir)
            return True
        except:
            return False

    def extract_single(self, file_name, output_dir):
        output_dir = Path(output_dir)
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as archive:
                archive.extract(file_name, output_dir)
            return True
        except:
            return False

    def zip_files(self, file_list, new_zip_name):
        new_zip_path = Path(new_zip_name)
        try:
            with zipfile.ZipFile(new_zip_path, 'w') as archive:
                for file in file_list:
                    archive.write(file)
            return True
        except:
            return False
