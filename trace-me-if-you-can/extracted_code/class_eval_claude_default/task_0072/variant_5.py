import re
from typing import List, Union

class RegexUtils:
    
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}-\d{3}-\d{4}\b'
    SENTENCE_SPLIT_PATTERN = r'[.!?][\s]{1,2}(?=[A-Z])'

    @staticmethod
    def _execute_regex_operation(operation: str, pattern: str, text: str, replacement: str = None) -> Union[bool, List[str], str]:
        operations = {
            'match': lambda p, t: bool(re.match(p, t)),
            'findall': lambda p, t: re.findall(p, t),
            'split': lambda p, t: re.split(p, t),
            'sub': lambda p, t: re.sub(p, replacement, t)
        }
        return operations[operation](pattern, text)

    def match(self, pattern: str, text: str) -> bool:
        return self._execute_regex_operation('match', pattern, text)

    def findall(self, pattern: str, text: str) -> List[str]:
        return self._execute_regex_operation('findall', pattern, text)

    def split(self, pattern: str, text: str) -> List[str]:
        return self._execute_regex_operation('split', pattern, text)

    def sub(self, pattern: str, replacement: str, text: str) -> str:
        return self._execute_regex_operation('sub', pattern, text, replacement)

    def generate_email_pattern(self) -> str:
        return self.EMAIL_PATTERN

    def generate_phone_number_pattern(self) -> str:
        return self.PHONE_PATTERN

    def generate_split_sentences_pattern(self) -> str:
        return self.SENTENCE_SPLIT_PATTERN

    def split_sentences(self, text: str) -> List[str]:
        return self.split(self.SENTENCE_SPLIT_PATTERN, text)

    def validate_phone_number(self, phone_number: str) -> bool:
        return self.match(self.PHONE_PATTERN, phone_number)

    def extract_email(self, text: str) -> List[str]:
        return self.findall(self.EMAIL_PATTERN, text)
