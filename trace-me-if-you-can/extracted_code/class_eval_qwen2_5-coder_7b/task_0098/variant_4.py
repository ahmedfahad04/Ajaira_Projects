import xml.etree.ElementTree as ET

class XMLProcessor:
    def __init__(self, file):
        self.file = file
        self.tree = None
        self.root = None

    def load_xml_data(self):
        try:
            self.tree = ET.parse(self.file)
            self.root = self.tree.getroot()
            return self.root
        except ET.ParseError:
            return None

    def save_xml_data(self, file_name):
        try:
            self.tree.write(file_name)
            return True
        except ET.WriteError:
            return False

    def update_xml_items(self, file_name):
        for item in self.root.findall('item'):
            item.text = item.text.upper()
        return self.save_xml_data(file_name)

    def get_xml_elements(self, element_name):
        return self.root.findall(element_name)
