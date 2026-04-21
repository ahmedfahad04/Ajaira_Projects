class TextNumberMapper:

    def __init__(self):
        self.number_map = {}
        self.digit_words = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        self.tens_words = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.scale_words = ["hundred", "thousand", "million", "billion", "trillion"]

        self.number_map["and"] = (1, 0)
        for idx, word in enumerate(self.digit_words):
            self.number_map[word] = (1, idx)
        for idx, word in enumerate(self.tens_words):
            self.number_map[word] = (1, idx * 10)
        for idx, word in enumerate(self.scale_words):
            self.number_map[word] = (10 ** (idx * 3 or 2), 0)

        self.ordinal_map = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        self.ordinal_patterns = [('ieth', 'y'), ('th', '')]

    def translate_text_to_numbers(self, text):
        text = text.replace('-', ' ')

        total = current = 0
        segment = ""
        number_found = False
        for word in text.split():
            if word in self.ordinal_map:
                scale, increment = (1, self.ordinal_map[word])
                current = current * scale + increment
                number_found = True
            else:
                for pattern, replacement in self.ordinal_patterns:
                    if word.endswith(pattern):
                        word = "%s%s" % (word[:-len(pattern)], replacement)

                if word not in self.number_map:
                    if number_found:
                        segment += repr(total + current) + " "
                    segment += word + " "
                    total = current = 0
                    number_found = False
                else:
                    scale, increment = self.number_map[word]
                    current = current * scale + increment
                    if scale > 100:
                        total += current
                        current = 0
                    number_found = True

        if number_found:
            segment += repr(total + current)

        return segment

    def validate_input(self, text):
        text = text.replace('-', ' ')

        for word in text.split():
            if word in self.ordinal_map:
                continue
            else:
                for pattern, replacement in self.ordinal_patterns:
                    if word.endswith(pattern):
                        word = "%s%s" % (word[:-len(pattern)], replacement)

                if word not in self.number_map:
                    return False

        return True
