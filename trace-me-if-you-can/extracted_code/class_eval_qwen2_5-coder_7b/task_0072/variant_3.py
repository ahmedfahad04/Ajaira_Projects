import re

class RegexFunctions:

    def match_pattern(self, pattern, text):
        return bool(re.match(pattern, text))

    def find_matches(self, pattern, text):
        return re.findall(pattern, text)

    def divide_text(self, pattern, text):
        return re.split(pattern, text)

    def substitute_text(self, pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def email_pattern(self):
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def phone_pattern(self):
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def sentence_pattern(self):
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def split_into_sentences(self, text):
        pattern = self.sentence_pattern()
        return self.divide_text(pattern, text)

    def validate_phone(self, phone_number):
        pattern = self.phone_pattern()
        return self.match_pattern(pattern, phone_number)

    def extract_emails(self, text):
        pattern = self.email_pattern()
        return self.find_matches(pattern, text)
