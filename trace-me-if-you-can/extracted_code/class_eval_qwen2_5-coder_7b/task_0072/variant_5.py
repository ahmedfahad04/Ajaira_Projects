import re

class RegexFunctions:

    def does_match(self, pattern, text):
        return bool(re.match(pattern, text))

    def find_all(self, pattern, text):
        return re.findall(pattern, text)

    def partition_text(self, pattern, text):
        return re.split(pattern, text)

    def replace_all(self, pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def email_pattern(self):
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def phone_pattern(self):
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def sentence_pattern(self):
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def split_into_sentences(self, text):
        pattern = self.sentence_pattern()
        return self.partition_text(pattern, text)

    def is_phone_number_valid(self, phone_number):
        pattern = self.phone_pattern()
        return self.does_match(pattern, phone_number)

    def extract_emails_from_text(self, text):
        pattern = self.email_pattern()
        return self.find_all(pattern, text)
