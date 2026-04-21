import re

class RegexHelper:

    def pattern_match(self, pattern, text):
        return bool(re.match(pattern, text))

    def find_all_matches(self, pattern, text):
        return re.findall(pattern, text)

    def split_text(self, pattern, text):
        return re.split(pattern, text)

    def replace_text(self, pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def create_email_pattern(self):
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def create_phone_pattern(self):
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def create_sentence_split_pattern(self):
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def split_into_sentences(self, text):
        pattern = self.create_sentence_split_pattern()
        return self.split_text(pattern, text)

    def check_phone_number(self, phone_number):
        pattern = self.create_phone_pattern()
        return self.pattern_match(pattern, phone_number)

    def extract_emails(self, text):
        pattern = self.create_email_pattern()
        return self.find_all_matches(pattern, text)
