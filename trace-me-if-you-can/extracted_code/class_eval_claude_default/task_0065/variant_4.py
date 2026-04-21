class NumberWordFormatter:
    DIGITS = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    TEENS = ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
    TENS = ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
    SCALES = ["", "THOUSAND", "MILLION", "BILLION"]

    def format(self, x):
        return self.format_string(str(x)) if x is not None else ""

    def format_string(self, x):
        integer_part, decimal_part = self._split_number(x)
        word_parts = self._convert_integer_to_words(integer_part)
        cents_part = self._format_cents(decimal_part)
        
        if not word_parts:
            return "ZERO ONLY"
        
        result = " ".join(word_parts)
        return f"{result} {cents_part}ONLY" if cents_part else f"{result} ONLY"

    def _split_number(self, x):
        parts = x.split('.') + ['']
        return parts[0], parts[1][:2] if parts[1] else ""

    def _convert_integer_to_words(self, integer_str):
        if not integer_str or integer_str == "0":
            return []
        
        # Process in groups of three digits from right to left
        groups = []
        padded = integer_str.zfill((len(integer_str) + 2) // 3 * 3)
        
        for i in range(0, len(padded), 3):
            group = padded[i:i+3]
            if group != "000":
                group_words = self._convert_three_digits(group)
                scale_index = (len(padded) - i) // 3 - 1
                scale = self.SCALES[scale_index] if scale_index < len(self.SCALES) else ""
                groups.append(f"{group_words} {scale}".strip())
        
        return groups

    def _convert_three_digits(self, three_digits):
        hundreds = int(three_digits[0])
        tens_units = three_digits[1:]
        
        parts = []
        if hundreds > 0:
            parts.append(f"{self.DIGITS[hundreds]} HUNDRED")
        
        tens_units_word = self._convert_two_digits(tens_units)
        if tens_units_word:
            if parts:
                parts.append(f"AND {tens_units_word}")
            else:
                parts.append(tens_units_word)
        
        return " ".join(parts)

    def _convert_two_digits(self, two_digits):
        two_digits = two_digits.zfill(2)
        tens, units = int(two_digits[0]), int(two_digits[1])
        
        if tens == 0:
            return self.DIGITS[units]
        elif tens == 1:
            return self.TEENS[int(two_digits) - 10]
        elif units == 0:
            return self.TENS[tens - 1]
        else:
            return f"{self.TENS[tens - 1]} {self.DIGITS[units]}"

    def _format_cents(self, decimal_part):
        return f"AND CENTS {self._convert_two_digits(decimal_part)} " if decimal_part else ""
