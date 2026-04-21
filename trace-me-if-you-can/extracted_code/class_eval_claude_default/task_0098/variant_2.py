import xml.etree.ElementTree as ET
from contextlib import contextmanager


class XMLProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.root = None

    @contextmanager
    def _safe_operation(self):
        try:
            yield
        except:
            pass

    def read_xml(self):
        with self._safe_operation():
            tree = ET.parse(self.file_name)
            self.root = tree.getroot()
            return self.root
        return None

    def write_xml(self, file_name):
        with self._safe_operation():
            tree = ET.ElementTree(self.root)
            tree.write(file_name)
            return True
        return False

    def process_xml_data(self, file_name):
        def transform_text(element):
            if element.text:
                element.text = element.text.upper()

        list(map(transform_text, self.root.iter('item')))
        return self.write_xml(file_name)

    def find_element(self, element_name):
        return list(filter(lambda e: e.tag == element_name, self.root.iter())) if self.root else []
