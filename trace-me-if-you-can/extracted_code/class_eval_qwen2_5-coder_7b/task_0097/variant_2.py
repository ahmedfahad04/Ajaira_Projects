class NumberTextConverter:

    def __init__(self):
        self.words_to_numbers = {}
        self.zero_to_nineteen = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        self.twenty_to_ninety = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.bigger_numbers = ["hundred", "thousand", "million", "billion", "trillion"]

        self.words_to_numbers["and"] = (1, 0)
        for idx, word in enumerate(self.zero_to_nineteen):
            self.words_to_numbers[word] = (1, idx)
        for idx, word in enumerate(self.twenty_to_ninety):
            self.words_to_numbers[word] = (1, idx * 10)
        for idx, word in enumerate(self.bigger_numbers):
            self.words_to_numbers[word] = (10 ** (idx * 3 or 2), 0)

        self.ordinal_numbers = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        self.ordinal_suffixes = [('ieth', 'y'), ('th', '')]

    def text_to_numeric(self, text):
        text = text.replace('-', ' ')

        cumulative_sum = current_value = 0
        temp_string = ""
        number_found = False
        for word in text.split():
            if word in self.ordinal_numbers:
                scale, increment = (1, self.ordinal_numbers[word])
                current_value = current_value * scale + increment
                number_found = True
            else:
                for suffix, replacement in self.ordinal_suffixes:
                    if word.endswith(suffix):
                        word = "%s%s" % (word[:-len(suffix)], replacement)

                if word not in self.words_to_numbers:
                    if number_found:
                        temp_string += repr(cumulative_sum + current_value) + " "
                    temp_string += word + " "
                    cumulative_sum = current_value = 0
                    number_found = False
                else:
                    scale, increment = self.words_to_numbers[word]
                    current_value = current_value * scale + increment
                    if scale > 100:
                        cumulative_sum += current_value
                        current_value = 0
                    number_found = True

        if number_found:
            temp_string += repr(cumulative_sum + current_value)

        return temp_string

    def is_input_valid(self, text):
        text = text.replace('-', ' ')

        for word in text.split():
            if word in self.ordinal_numbers:
                continue
            else:
                for suffix, replacement in self.ordinal_suffixes:
                    if word.endswith(suffix):
                        word = "%s%s" % (word[:-len(suffix)], replacement)

                if word not in self.words_to_numbers:
                    return False

        return True
