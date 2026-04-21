import xml.etree.ElementTree as ET
from pathlib import Path


class XMLProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self._tree = None

    @property
    def root(self):
        return self._tree.getroot() if self._tree else None

    def read_xml(self):
        try:
            self._tree = ET.parse(self.file_name)
            return self.root
        except:
            return None

    def write_xml(self, file_name):
        if not self._tree:
            return False
        try:
            self._tree.write(file_name)
            return True
        except:
            return False

    def process_xml_data(self, file_name):
        if not self.root:
            return False
        
        items = self.root.iter('item')
        for element in items:
            if element.text:
                element.text = element.text.upper()
        return self.write_xml(file_name)

    def find_element(self, element_name):
        return self.root.findall(element_name) if self.root else []
