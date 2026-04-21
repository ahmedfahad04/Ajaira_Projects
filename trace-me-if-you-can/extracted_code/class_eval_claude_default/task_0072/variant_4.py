import re

class RegexUtils:
    
    def __init__(self):
        self._compiled_patterns = {}
    
    def _get_compiled_pattern(self, pattern_name):
        if pattern_name not in self._compiled_patterns:
            pattern_map = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}-\d{3}-\d{4}\b',
                'sentence_split': r'[.!?][\s]{1,2}(?=[A-Z])'
            }
            self._compiled_patterns[pattern_name] = re.compile(pattern_map[pattern_name])
        return self._compiled_patterns[pattern_name]

    def match(self, pattern, text):
        if isinstance(pattern, str):
            compiled_pattern = re.compile(pattern)
        else:
            compiled_pattern = pattern
        return compiled_pattern.match(text) is not None

    def findall(self, pattern, text):
        if isinstance(pattern, str):
            return re.findall(pattern, text)
        return pattern.findall(text)

    def split(self, pattern, text):
        if isinstance(pattern, str):
            return re.split(pattern, text)
        return pattern.split(text)

    def sub(self, pattern, replacement, text):
        if isinstance(pattern, str):
            return re.sub(pattern, replacement, text)
        return pattern.sub(replacement, text)

    def generate_email_pattern(self):
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def generate_phone_number_pattern(self):
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def generate_split_sentences_pattern(self):
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def split_sentences(self, text):
        pattern = self._get_compiled_pattern('sentence_split')
        return self.split(pattern, text)

    def validate_phone_number(self, phone_number):
        pattern = self._get_compiled_pattern('phone')
        return self.match(pattern, phone_number)

    def extract_email(self, text):
        pattern = self._get_compiled_pattern('email')
        return self.findall(pattern, text)
