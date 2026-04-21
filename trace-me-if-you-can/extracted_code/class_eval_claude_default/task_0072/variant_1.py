import re
from functools import partial

class RegexUtils:
    
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b',
        'sentence_split': r'[.!?][\s]{1,2}(?=[A-Z])'
    }
    
    def __init__(self):
        self._regex_match = partial(re.match)
        self._regex_findall = partial(re.findall)
        self._regex_split = partial(re.split)
        self._regex_sub = partial(re.sub)

    def match(self, pattern, text):
        return bool(self._regex_match(pattern, text))

    def findall(self, pattern, text):
        return self._regex_findall(pattern, text)

    def split(self, pattern, text):
        return self._regex_split(pattern, text)

    def sub(self, pattern, replacement, text):
        return self._regex_sub(pattern, replacement, text)

    def generate_email_pattern(self):
        return self.PATTERNS['email']

    def generate_phone_number_pattern(self):
        return self.PATTERNS['phone']

    def generate_split_sentences_pattern(self):
        return self.PATTERNS['sentence_split']

    def split_sentences(self, text):
        return self.split(self.PATTERNS['sentence_split'], text)

    def validate_phone_number(self, phone_number):
        return self.match(self.PATTERNS['phone'], phone_number)

    def extract_email(self, text):
        return self.findall(self.PATTERNS['email'], text)
