import re

class RegexTools:

    def is_match(self, pattern, text):
        return bool(re.match(pattern, text))

    def all_matches(self, pattern, text):
        return re.findall(pattern, text)

    def segment_text(self, pattern, text):
        return re.split(pattern, text)

    def change_text(self, pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def get_email_pattern(self):
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def get_phone_pattern(self):
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def get_sentence_pattern(self):
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def sentence_segments(self, text):
        pattern = self.get_sentence_pattern()
        return self.segment_text(pattern, text)

    def is_valid_phone(self, phone_number):
        pattern = self.get_phone_pattern()
        return self.is_match(pattern, phone_number)

    def emails_in_text(self, text):
        pattern = self.get_email_pattern()
        return self.all_matches(pattern, text)
