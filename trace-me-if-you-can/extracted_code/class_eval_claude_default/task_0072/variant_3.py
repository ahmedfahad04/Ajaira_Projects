import re

def create_regex_utils():
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b',
        'sentence_split': r'[.!?][\s]{1,2}(?=[A-Z])'
    }
    
    def match(pattern, text):
        match_obj = re.match(pattern, text)
        return match_obj is not None

    def findall(pattern, text):
        return re.findall(pattern, text)

    def split(pattern, text):
        return re.split(pattern, text)

    def sub(pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def generate_email_pattern():
        return patterns['email']

    def generate_phone_number_pattern():
        return patterns['phone']

    def generate_split_sentences_pattern():
        return patterns['sentence_split']

    def split_sentences(text):
        return split(generate_split_sentences_pattern(), text)

    def validate_phone_number(phone_number):
        return match(generate_phone_number_pattern(), phone_number)

    def extract_email(text):
        return findall(generate_email_pattern(), text)
    
    return type('RegexUtils', (), {
        'match': match,
        'findall': findall,
        'split': split,
        'sub': sub,
        'generate_email_pattern': generate_email_pattern,
        'generate_phone_number_pattern': generate_phone_number_pattern,
        'generate_split_sentences_pattern': generate_split_sentences_pattern,
        'split_sentences': split_sentences,
        'validate_phone_number': validate_phone_number,
        'extract_email': extract_email
    })()

RegexUtils = create_regex_utils()
