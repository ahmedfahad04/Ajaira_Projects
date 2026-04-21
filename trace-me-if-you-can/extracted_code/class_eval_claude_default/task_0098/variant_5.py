import xml.etree.ElementTree as ET


class XMLProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.root = None
        self._operations = {
            'read': self._read_operation,
            'write': self._write_operation,
            'process': self._process_operation,
            'find': self._find_operation
        }

    def _read_operation(self, *args):
        try:
            tree = ET.parse(self.file_name)
            self.root = tree.getroot()
            return self.root
        except:
            return None

    def _write_operation(self, file_name):
        try:
            tree = ET.ElementTree(self.root)
            tree.write(file_name)
            return True
        except:
            return False

    def _process_operation(self, file_name):
        items = [elem for elem in self.root.iter('item')]
        
        for idx in range(len(items)):
            current_text = items[idx].text
            items[idx].text = current_text.upper() if current_text else current_text
            
        return self._write_operation(file_name)

    def _find_operation(self, element_name):
        return self.root.findall(element_name) if self.root else []

    def read_xml(self):
        return self._operations['read']()

    def write_xml(self, file_name):
        return self._operations['write'](file_name)

    def process_xml_data(self, file_name):
        return self._operations['process'](file_name)

    def find_element(self, element_name):
        return self._operations['find'](element_name)
