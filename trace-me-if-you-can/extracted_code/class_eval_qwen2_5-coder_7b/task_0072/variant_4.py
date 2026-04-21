import re

class RegexHelpers:

    def check_match(self, pattern, text):
        return bool(re.match(pattern, text))

    def get_all_matches(self, pattern, text):
        return re.findall(pattern, text)

    def text_split(self, pattern, text):
        return re.split(pattern, text)

    def substitute(self, pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def email_regex(self):
        return r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def phone_regex(self):
        return r'\b\d{3}-\d{3}-\d{4}\b'

    def sentence_regex(self):
        return r'[.!?][\s]{1,2}(?=[A-Z])'

    def split_sentences(self, text):
        pattern = self.sentence_regex()
        return self.text_split(pattern, text)

    def is_phone_valid(self, phone_number):
        pattern = self.phone_regex()
        return self.check_match(pattern, phone_number)

    def get_emails(self, text):
        pattern = self.email_regex()
        return self.get_all_matches(pattern, text)
