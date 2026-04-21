import xml.etree.ElementTree as ET

class XMLHandler:
    def __init__(self, xml_file):
        self.file_path = xml_file
        self.tree = None
        self.root = None

    def load_xml(self):
        try:
            self.tree = ET.parse(self.file_path)
            self.root = self.tree.getroot()
            return self.root
        except ET.ParseError:
            return None

    def save_xml(self, output_file):
        try:
            self.tree.write(output_file)
            return True
        except ET.WriteError:
            return False

    def modify_xml_content(self, output_file):
        for item in self.root.iter('item'):
            item.text = item.text.upper()
        return self.save_xml(output_file)

    def locate_elements(self, tag_name):
        return self.root.findall(tag_name)
