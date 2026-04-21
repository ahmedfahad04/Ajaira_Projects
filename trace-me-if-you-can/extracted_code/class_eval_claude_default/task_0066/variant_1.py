import re

class NumericEntityUnescaper:
    def __init__(self):
        self.pattern = re.compile(r'&#(?:x([0-9a-fA-F]+)|([0-9]+));')

    def replace(self, string):
        def convert_match(match):
            hex_digits, decimal_digits = match.groups()
            if hex_digits:
                return chr(int(hex_digits, 16))
            else:
                return chr(int(decimal_digits, 10))
        
        try:
            return self.pattern.sub(convert_match, string)
        except (ValueError, OverflowError):
            return string
