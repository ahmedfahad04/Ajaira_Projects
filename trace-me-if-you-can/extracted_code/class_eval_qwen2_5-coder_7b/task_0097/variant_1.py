class NumericWords:

    def __init__(self):
        self.numbers = {}
        self.single_digits = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        self.decades = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.large_numbers = ["hundred", "thousand", "million", "billion", "trillion"]

        self.numbers["and"] = (1, 0)
        for idx, word in enumerate(self.single_digits):
            self.numbers[word] = (1, idx)
        for idx, word in enumerate(self.decades):
            self.numbers[word] = (1, idx * 10)
        for idx, word in enumerate(self.large_numbers):
            self.numbers[word] = (10 ** (idx * 3 or 2), 0)

        self.ordinal_numbers = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        self.ordinal_suffixes = [('ieth', 'y'), ('th', '')]

    def convert_text_to_number(self, text):
        text = text.replace('-', ' ')

        total = current = 0
        segment = ""
        found_number = False
        for word in text.split():
            if word in self.ordinal_numbers:
                scale, increment = (1, self.ordinal_numbers[word])
                current = current * scale + increment
                found_number = True
            else:
                for suffix, replacement in self.ordinal_suffixes:
                    if word.endswith(suffix):
                        word = "%s%s" % (word[:-len(suffix)], replacement)

                if word not in self.numbers:
                    if found_number:
                        segment += repr(total + current) + " "
                    segment += word + " "
                    total = current = 0
                    found_number = False
                else:
                    scale, increment = self.numbers[word]
                    current = current * scale + increment
                    if scale > 100:
                        total += current
                        current = 0
                    found_number = True

        if found_number:
            segment += repr(total + current)

        return segment

    def check_valid_input(self, text):
        text = text.replace('-', ' ')

        for word in text.split():
            if word in self.ordinal_numbers:
                continue
            else:
                for suffix, replacement in self.ordinal_suffixes:
                    if word.endswith(suffix):
                        word = "%s%s" % (word[:-len(suffix)], replacement)

                if word not in self.numbers:
                    return False

        return True
