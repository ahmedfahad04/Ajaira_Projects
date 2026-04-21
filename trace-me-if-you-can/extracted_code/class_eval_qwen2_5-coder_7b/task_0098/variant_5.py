import xml.etree.ElementTree as ET

class XMLModifier:
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.tree = None
        self.root = None

    def import_xml(self):
        try:
            self.tree = ET.parse(self.xml_file)
            self.root = self.tree.getroot()
            return self.root
        except ET.ParseError:
            return None

    def export_xml(self, output_file):
        try:
            self.tree.write(output_file)
            return True
        except ET.WriteError:
            return False

    def modify_items(self, output_file):
        for item in self.root.findall('item'):
            item.text = item.text.upper()
        return self.export_xml(output_file)

    def find_elements(self, element_tag):
        return self.root.findall(element_tag)
