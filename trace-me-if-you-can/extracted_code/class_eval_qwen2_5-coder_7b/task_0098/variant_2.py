import xml.etree.ElementTree as ET

class XMLManager:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.tree = None
        self.root = None

    def parse_xml(self):
        try:
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()
            return self.root
        except ET.ParseError:
            return None

    def export_xml(self, output_path):
        try:
            self.tree.write(output_path)
            return True
        except ET.WriteError:
            return False

    def update_xml_data(self, output_path):
        for item in self.root.findall('.//item'):
            item.text = item.text.upper()
        return self.export_xml(output_path)

    def search_xml(self, tag):
        return self.root.findall(tag)
