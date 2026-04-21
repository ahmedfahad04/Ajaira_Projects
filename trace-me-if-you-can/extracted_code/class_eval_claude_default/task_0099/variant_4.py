import zipfile


class ZipFileProcessor:
    SUCCESS = True
    FAILURE = False
    
    def __init__(self, file_name):
        self.file_name = file_name
        self._operations = {
            'extract_all': lambda zf, *args: zf.extractall(*args),
            'extract_file': lambda zf, *args: zf.extract(*args)
        }

    def read_zip_file(self):
        result = self._safe_execute(lambda: zipfile.ZipFile(self.file_name, 'r'))
        return result if result is not None else None

    def extract_all(self, output_path):
        return self._execute_zip_operation('extract_all', output_path)

    def extract_file(self, file_name, output_path):
        return self._execute_zip_operation('extract_file', file_name, output_path)

    def create_zip_file(self, files, output_file_name):
        def create_operation():
            with zipfile.ZipFile(output_file_name, 'w') as zip_file:
                [zip_file.write(file) for file in files]
            return self.SUCCESS
        
        return self._safe_execute(create_operation) or self.FAILURE

    def _execute_zip_operation(self, operation_name, *args):
        def zip_operation():
            with zipfile.ZipFile(self.file_name, 'r') as zip_file:
                self._operations[operation_name](zip_file, *args)
            return self.SUCCESS
        
        return self._safe_execute(zip_operation) or self.FAILURE

    def _safe_execute(self, operation):
        try:
            return operation()
        except:
            return None
