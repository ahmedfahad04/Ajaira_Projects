import xml.etree.ElementTree as ET

class XMLUtility:
    def __init__(self, file_location):
        self.file_location = file_location
        self.xml_tree = None
        self.root_node = None

    def load_xml_file(self):
        try:
            self.xml_tree = ET.parse(self.file_location)
            self.root_node = self.xml_tree.getroot()
            return self.root_node
        except ET.ParseError:
            return None

    def write_xml_file(self, output_file):
        try:
            self.xml_tree.write(output_file)
            return True
        except ET.WriteError:
            return False

    def process_xml(self, output_file):
        for item in self.root_node.findall('item'):
            item.text = item.text.upper()
        return self.write_xml_file(output_file)

    def search_elements(self, element_tag):
        return self.root_node.findall(element_tag)
