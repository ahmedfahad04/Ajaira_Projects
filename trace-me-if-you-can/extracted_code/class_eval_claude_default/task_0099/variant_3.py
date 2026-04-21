import zipfile
from functools import wraps


def handle_zip_exceptions(default_return=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                return default_return
        return wrapper
    return decorator


class ZipFileProcessor:
    def __init__(self, file_name):
        self.file_name = file_name

    @handle_zip_exceptions(default_return=None)
    def read_zip_file(self):
        return zipfile.ZipFile(self.file_name, 'r')

    @handle_zip_exceptions(default_return=False)
    def extract_all(self, output_path):
        with zipfile.ZipFile(self.file_name, 'r') as zip_file:
            zip_file.extractall(output_path)
        return True

    @handle_zip_exceptions(default_return=False)
    def extract_file(self, file_name, output_path):
        with zipfile.ZipFile(self.file_name, 'r') as zip_file:
            zip_file.extract(file_name, output_path)
        return True

    @handle_zip_exceptions(default_return=False)
    def create_zip_file(self, files, output_file_name):
        with zipfile.ZipFile(output_file_name, 'w') as zip_file:
            for file in files:
                zip_file.write(file)
        return True
