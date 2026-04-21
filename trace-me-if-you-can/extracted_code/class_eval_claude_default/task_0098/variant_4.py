import xml.etree.ElementTree as ET
from typing import Optional, List


class XMLProcessor:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.root: Optional[ET.Element] = None

    def read_xml(self) -> Optional[ET.Element]:
        parser_result = self._parse_file()
        if parser_result:
            self.root = parser_result
        return self.root

    def _parse_file(self) -> Optional[ET.Element]:
        try:
            return ET.parse(self.file_name).getroot()
        except Exception:
            return None

    def write_xml(self, file_name: str) -> bool:
        return self._write_to_file(file_name)

    def _write_to_file(self, file_name: str) -> bool:
        try:
            ET.ElementTree(self.root).write(file_name)
            return True
        except Exception:
            return False

    def process_xml_data(self, file_name: str) -> bool:
        self._transform_items()
        return self.write_xml(file_name)

    def _transform_items(self) -> None:
        item_elements = self._get_items()
        self._uppercase_text(item_elements)

    def _get_items(self) -> List[ET.Element]:
        return list(self.root.iter('item'))

    def _uppercase_text(self, elements: List[ET.Element]) -> None:
        for element in elements:
            if element.text:
                element.text = element.text.upper()

    def find_element(self, element_name: str) -> List[ET.Element]:
        return self.root.findall(element_name) if self.root else []
