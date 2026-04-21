import zipfile
from pathlib import Path


class ZipFileProcessor:
    def __init__(self, file_name):
        self.file_path = Path(file_name)

    def read_zip_file(self):
        if not self.file_path.exists():
            return None
        try:
            return zipfile.ZipFile(str(self.file_path), 'r')
        except zipfile.BadZipFile:
            return None
        except (OSError, IOError):
            return None

    def extract_all(self, output_path):
        return self._perform_zip_operation(
            lambda zf: zf.extractall(output_path)
        )

    def extract_file(self, file_name, output_path):
        return self._perform_zip_operation(
            lambda zf: zf.extract(file_name, output_path)
        )

    def create_zip_file(self, files, output_file_name):
        try:
            with zipfile.ZipFile(output_file_name, 'w') as zip_file:
                for file in files:
                    zip_file.write(file)
            return True
        except:
            return False

    def _perform_zip_operation(self, operation):
        try:
            with zipfile.ZipFile(str(self.file_path), 'r') as zip_file:
                operation(zip_file)
            return True
        except:
            return False
