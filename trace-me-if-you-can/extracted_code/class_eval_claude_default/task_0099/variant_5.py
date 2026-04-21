import zipfile


class ZipOperationResult:
    def __init__(self, success, data=None):
        self.success = success
        self.data = data
    
    def __bool__(self):
        return self.success


class ZipFileProcessor:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_zip_file(self):
        result = self._try_operation(lambda: zipfile.ZipFile(self.file_name, 'r'))
        return result.data if result.success else None

    def extract_all(self, output_path):
        result = self._try_zip_context_operation(
            lambda zip_file: zip_file.extractall(output_path)
        )
        return bool(result)

    def extract_file(self, file_name, output_path):
        result = self._try_zip_context_operation(
            lambda zip_file: zip_file.extract(file_name, output_path)
        )
        return bool(result)

    def create_zip_file(self, files, output_file_name):
        def write_files():
            with zipfile.ZipFile(output_file_name, 'w') as zip_file:
                for file in files:
                    zip_file.write(file)
        
        result = self._try_operation(write_files)
        return bool(result)

    def _try_operation(self, operation):
        try:
            data = operation()
            return ZipOperationResult(True, data)
        except:
            return ZipOperationResult(False)

    def _try_zip_context_operation(self, operation):
        def context_operation():
            with zipfile.ZipFile(self.file_name, 'r') as zip_file:
                operation(zip_file)
        
        return self._try_operation(context_operation)
