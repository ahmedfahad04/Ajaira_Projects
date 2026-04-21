class NumberConverter:

    def __init__(self):
        self.number_word_map = {}
        self.one_through_nineteen = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.large_scales = ["hundred", "thousand", "million", "billion", "trillion"]

        self.number_word_map["and"] = (1, 0)
        for idx, word in enumerate(self.one_through_nineteen):
            self.number_word_map[word] = (1, idx)
        for idx, word in enumerate(self.tens):
            self.number_word_map[word] = (1, idx * 10)
        for idx, word in enumerate(self.large_scales):
            self.number_word_map[word] = (10 ** (idx * 3 or 2), 0)

        self.ordinal_numbers = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        self.ordinal_suffixes = [('ieth', 'y'), ('th', '')]

    def convert_text_to_number(self, text):
        text = text.replace('-', ' ')

        total = current = 0
        segment = ""
        number_found = False
        for word in text.split():
            if word in self.ordinal_numbers:
                scale, increment = (1, self.ordinal_numbers[word])
                current = current * scale + increment
                number_found = True
            else:
                for suffix, replacement in self.ordinal_suffixes:
                    if word.endswith(suffix):
                        word = "%s%s" % (word[:-len(suffix)], replacement)

                if word not in self.number_word_map:
                    if number_found:
                        segment += repr(total + current) + " "
                    segment += word + " "
                    total = current = 0
                    number_found = False
                else:
                    scale, increment = self.number_word_map[word]
                    current = current * scale + increment
                    if scale > 100:
                        total += current
                        current = 0
                    number_found = True

        if number_found:
            segment += repr(total + current)

        return segment

    def check_input_validity(self, text):
        text = text.replace('-', ' ')

        for word in text.split():
            if word in self.ordinal_numbers:
                continue
            else:
                for suffix, replacement in self.ordinal_suffixes:
                    if word.endswith(suffix):
                        word = "%s%s" % (word[:-len(suffix)], replacement)

                if word not in self.number_word_map:
                    return False

        return True
