import xml.etree.ElementTree as ET


class XMLFileHandler:
    @staticmethod
    def load_xml(file_name):
        try:
            return ET.parse(file_name).getroot()
        except:
            return None

    @staticmethod
    def save_xml(root, file_name):
        try:
            ET.ElementTree(root).write(file_name)
            return True
        except:
            return False


class XMLProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.root = None

    def read_xml(self):
        self.root = XMLFileHandler.load_xml(self.file_name)
        return self.root

    def write_xml(self, file_name):
        return XMLFileHandler.save_xml(self.root, file_name)

    def process_xml_data(self, file_name):
        element_processor = lambda elem: setattr(elem, 'text', elem.text.upper()) if elem.text else None
        
        for element in self.root.iter('item'):
            element_processor(element)
        
        return self.write_xml(file_name)

    def find_element(self, element_name):
        xpath_query = f".//{element_name}"
        return self.root.findall(xpath_query) if self.root else []
