import re
from abc import ABC, abstractmethod

class PatternGenerator(ABC):
    @abstractmethod
    def get_pattern(self):
        pass

class EmailPattern(PatternGenerator):
    def get_pattern(self):
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

class PhonePattern(PatternGenerator):
    def get_pattern(self):
        return r'\b\d{3}-\d{3}-\d{4}\b'

class SentenceSplitPattern(PatternGenerator):
    def get_pattern(self):
        return r'[.!?][\s]{1,2}(?=[A-Z])'

class RegexUtils:
    
    def __init__(self):
        self._email_gen = EmailPattern()
        self._phone_gen = PhonePattern()
        self._sentence_gen = SentenceSplitPattern()

    def match(self, pattern, text):
        result = re.match(pattern, text)
        return result is not None

    def findall(self, pattern, text):
        return re.findall(pattern, text)

    def split(self, pattern, text):
        return re.split(pattern, text)

    def sub(self, pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def generate_email_pattern(self):
        return self._email_gen.get_pattern()

    def generate_phone_number_pattern(self):
        return self._phone_gen.get_pattern()

    def generate_split_sentences_pattern(self):
        return self._sentence_gen.get_pattern()

    def split_sentences(self, text):
        pattern = self.generate_split_sentences_pattern()
        return self.split(pattern, text)

    def validate_phone_number(self, phone_number):
        pattern = self.generate_phone_number_pattern()
        return self.match(pattern, phone_number)

    def extract_email(self, text):
        pattern = self.generate_email_pattern()
        return self.findall(pattern, text)
