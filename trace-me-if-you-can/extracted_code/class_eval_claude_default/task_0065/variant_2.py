class NumberWordFormatter:
    def __init__(self):
        self._setup_lookup_tables()

    def _setup_lookup_tables(self):
        self.word_mappings = {
            'ones': ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"],
            'teens': ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"],
            'tens': ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"],
            'scales': ["", "THOUSAND", "MILLION", "BILLION"]
        }

    def format(self, x):
        if x is None:
            return ""
        
        number_str = str(x)
        integer_part, fractional_part = self._split_number(number_str)
        
        if not integer_part or integer_part == "0":
            return "ZERO ONLY"
        
        main_words = self._convert_integer_to_words(integer_part)
        cents_words = self._format_cents(fractional_part)
        
        return f"{main_words} {cents_words}ONLY"

    def _split_number(self, number_str):
        if '.' in number_str:
            parts = number_str.split('.')
            return parts[0], parts[1]
        return number_str, ""

    def _convert_integer_to_words(self, integer_str):
        # Process number in groups of three digits
        digit_groups = []
        temp = integer_str[::-1]  # Reverse for easier processing
        
        while len(temp) > 0:
            group = temp[:3][::-1]  # Take up to 3 digits and reverse back
            digit_groups.append(group.zfill(3))
            temp = temp[3:]
        
        words = []
        for i, group in enumerate(digit_groups):
            if int(group) > 0:
                group_words = self._three_digit_group_to_words(group)
                scale_word = self.word_mappings['scales'][i] if i < len(self.word_mappings['scales']) else ""
                if scale_word:
                    words.append(f"{group_words} {scale_word}")
                else:
                    words.append(group_words)
        
        return " ".join(reversed(words))

    def _format_cents(self, fractional_part):
        return f"AND CENTS {self._two_digits_to_words(fractional_part)} " if fractional_part else ""

    def _two_digits_to_words(self, digits_str):
        padded = digits_str.ljust(2, '0')[:2]
        tens, ones = int(padded[0]), int(padded[1])
        
        if tens == 0:
            return self.word_mappings['ones'][ones]
        elif tens == 1:
            return self.word_mappings['teens'][int(padded) - 10]
        elif ones == 0:
            return self.word_mappings['tens'][tens - 1]
        else:
            return f"{self.word_mappings['tens'][tens - 1]} {self.word_mappings['ones'][ones]}"

    def _three_digit_group_to_words(self, three_digits):
        hundreds = int(three_digits[0])
        tens_ones = three_digits[1:]
        
        result = ""
        if hundreds > 0:
            result = f"{self.word_mappings['ones'][hundreds]} HUNDRED"
            if tens_ones != "00":
                result += f" AND {self._two_digits_to_words(tens_ones)}"
        else:
            result = self._two_digits_to_words(tens_ones)
        
        return result
