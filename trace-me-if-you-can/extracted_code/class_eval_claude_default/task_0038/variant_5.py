import openpyxl
from typing import List, Tuple, Optional, Union


class ExcelProcessor:
    def __init__(self):
        pass

    def read_excel(self, file_name: str) -> Optional[List[Tuple]]:
        workbook = None
        try:
            workbook = openpyxl.load_workbook(file_name)
            sheet = workbook.active
            data = []
            for row in sheet.rows:
                data.append(tuple(cell.value for cell in row))
            return data
        except Exception:
            return None
        finally:
            if workbook:
                workbook.close()

    def write_excel(self, data: List[Union[Tuple, List]], file_name: str) -> int:
        workbook = None
        try:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            for row_data in data:
                sheet.append(list(row_data))
            workbook.save(file_name)
            return 1
        except Exception:
            return 0
        finally:
            if workbook:
                workbook.close()

    def _generate_output_filename(self, input_filename: str) -> str:
        name_parts = input_filename.split('.')
        return name_parts[0] + '_process.xlsx'

    def _process_cell_value(self, value) -> Union[str, any]:
        str_value = str(value)
        return value if str_value.isdigit() else str_value.upper()

    def process_excel_data(self, N: int, save_file_name: str) -> Union[int, Tuple[int, str]]:
        data = self.read_excel(save_file_name)
        
        # Validation
        if data is None:
            return 0
        if not data or N >= len(data[0]):
            return 0
        
        # Processing
        processed_rows = []
        for original_row in data:
            processed_value = self._process_cell_value(original_row[N])
            new_row = list(original_row) + [processed_value]
            processed_rows.append(new_row)
        
        # Output
        output_filename = self._generate_output_filename(save_file_name)
        write_success = self.write_excel(processed_rows, output_filename)
        return write_success, output_filename
