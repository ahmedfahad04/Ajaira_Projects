class NumberWordFormatter:
    DIGITS = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    TEENS = ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
    TENS = ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
    SCALE_WORDS = ["", "THOUSAND", "MILLION", "BILLION"]

    def format(self, x):
        return self._convert_number_string(str(x)) if x is not None else ""

    def _convert_number_string(self, x):
        parts = x.split('.')
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else ""
        
        if not integer_part or integer_part == "0":
            main_text = "ZERO"
        else:
            main_text = self._process_integer_part(integer_part)
        
        cents_text = f"AND CENTS {self._convert_two_digits(decimal_part)} " if decimal_part else ""
        
        return f"{main_text} {cents_text}ONLY" if main_text != "ZERO" else "ZERO ONLY"

    def _process_integer_part(self, num_str):
        # Group digits into chunks of 3 from right to left
        chunks = []
        for i in range(len(num_str), 0, -3):
            start = max(0, i - 3)
            chunks.append(num_str[start:i])
        
        result_parts = []
        for idx, chunk in enumerate(reversed(chunks)):
            if chunk != "000" and int(chunk) != 0:
                chunk_text = self._convert_three_digits(chunk)
                scale_word = self.SCALE_WORDS[idx] if idx < len(self.SCALE_WORDS) else ""
                if scale_word:
                    result_parts.append(f"{chunk_text} {scale_word}")
                else:
                    result_parts.append(chunk_text)
        
        return " ".join(result_parts)

    def _convert_two_digits(self, s):
        s = s.ljust(2, '0')[:2]
        tens_digit, ones_digit = int(s[0]), int(s[1])
        
        if tens_digit == 0:
            return self.DIGITS[ones_digit]
        elif tens_digit == 1:
            return self.TEENS[int(s) - 10]
        elif ones_digit == 0:
            return self.TENS[tens_digit - 1]
        else:
            return f"{self.TENS[tens_digit - 1]} {self.DIGITS[ones_digit]}"

    def _convert_three_digits(self, s):
        s = s.zfill(3)
        hundreds_digit = int(s[0])
        remaining_two = s[1:]
        
        if hundreds_digit == 0:
            return self._convert_two_digits(remaining_two)
        elif remaining_two == "00":
            return f"{self.DIGITS[hundreds_digit]} HUNDRED"
        else:
            return f"{self.DIGITS[hundreds_digit]} HUNDRED AND {self._convert_two_digits(remaining_two)}"
