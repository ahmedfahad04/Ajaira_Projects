import zipfile
from pathlib import Path

class ArchiveManager:
    def __init__(self, archive_path):
        self.archive_path = Path(archive_path)

    def open_archive(self):
        try:
            return zipfile.ZipFile(self.archive_path, 'r')
        except:
            return None

    def extract_everything(self, destination_folder):
        destination_folder = Path(destination_folder)
        try:
            with zipfile.ZipFile(self.archive_path, 'r') as archive:
                archive.extractall(destination_folder)
            return True
        except:
            return False

    def extract_specific(self, file_name, destination_folder):
        destination_folder = Path(destination_folder)
        try:
            with zipfile.ZipFile(self.archive_path, 'r') as archive:
                archive.extract(file_name, destination_folder)
            return True
        except:
            return False

    def compile_archive(self, source_files, output_name):
        output_path = Path(output_name)
        try:
            with zipfile.ZipFile(output_path, 'w') as archive:
                for file in source_files:
                    archive.write(file)
            return True
        except:
            return False
