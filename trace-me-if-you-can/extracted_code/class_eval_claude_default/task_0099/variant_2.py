import zipfile
from contextlib import contextmanager


class ZipFileProcessor:
    def __init__(self, file_name):
        self.file_name = file_name

    @contextmanager
    def _zip_context(self, mode='r'):
        zip_file = None
        try:
            zip_file = zipfile.ZipFile(self.file_name, mode)
            yield zip_file
        except:
            yield None
        finally:
            if zip_file:
                zip_file.close()

    def read_zip_file(self):
        with self._zip_context('r') as zip_file:
            return zip_file

    def extract_all(self, output_path):
        with self._zip_context('r') as zip_file:
            if zip_file is None:
                return False
            try:
                zip_file.extractall(output_path)
                return True
            except:
                return False

    def extract_file(self, file_name, output_path):
        with self._zip_context('r') as zip_file:
            if zip_file is None:
                return False
            try:
                zip_file.extract(file_name, output_path)
                return True
            except:
                return False

    def create_zip_file(self, files, output_file_name):
        try:
            with zipfile.ZipFile(output_file_name, 'w') as zip_file:
                list(map(zip_file.write, files))
            return True
        except:
            return False
