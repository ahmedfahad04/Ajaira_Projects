class NumberWordFormatter:
    def __init__(self):
        self.word_maps = {
            'units': ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"],
            'teens': ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"],
            'tens': ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"],
            'scales': ["", "THOUSAND", "MILLION", "BILLION"]
        }

    def format(self, x):
        if x is None:
            return ""
        
        num_str = str(x)
        main_part, fractional_part = self._parse_input(num_str)
        
        if main_part == "0" or not main_part:
            main_words = ""
        else:
            main_words = self._process_integer_part(main_part)
        
        fractional_words = self._process_fractional_part(fractional_part)
        
        return self._assemble_final_result(main_words, fractional_words)

    def _parse_input(self, num_str):
        components = num_str.split('.')
        main = components[0] if components else ""
        fractional = components[1] if len(components) > 1 else ""
        return main, fractional

    def _process_integer_part(self, integer_str):
        reversed_digits = integer_str[::-1]
        padded_digits = self._pad_to_multiple_of_three(reversed_digits)
        
        result_segments = []
        chunk_count = len(padded_digits) // 3
        
        for chunk_idx in range(chunk_count):
            start_pos = chunk_idx * 3
            chunk = padded_digits[start_pos:start_pos + 3][::-1]
            
            if chunk != "000":
                chunk_words = self._convert_chunk_to_words(chunk)
                scale_word = self.word_maps['scales'][chunk_idx] if chunk_idx < len(self.word_maps['scales']) else ""
                segment = f"{chunk_words} {scale_word}".strip()
                result_segments.insert(0, segment)
        
        return " ".join(result_segments)

    def _pad_to_multiple_of_three(self, digits):
        remainder = len(digits) % 3
        if remainder == 1:
            return digits + "00"
        elif remainder == 2:
            return digits + "0"
        return digits

    def _convert_chunk_to_words(self, chunk):
        hundreds_digit = int(chunk[0])
        tens_and_units = chunk[1:]
        
        words = []
        
        if hundreds_digit != 0:
            words.append(f"{self.word_maps['units'][hundreds_digit]} HUNDRED")
        
        two_digit_words = self._handle_two_digits(tens_and_units)
        if two_digit_words:
            connector = "AND " if words else ""
            words.append(f"{connector}{two_digit_words}")
        
        return " ".join(words)

    def _handle_two_digits(self, digits):
        digits = digits.zfill(2)
        tens_digit, units_digit = int(digits[0]), int(digits[1])
        
        if tens_digit == 0:
            return self.word_maps['units'][units_digit]
        elif tens_digit == 1:
            return self.word_maps['teens'][int(digits) - 10]
        elif units_digit == 0:
            return self.word_maps['tens'][tens_digit - 1]
        else:
            return f"{self.word_maps['tens'][tens_digit - 1]} {self.word_maps['units'][units_digit]}"

    def _process_fractional_part(self, fractional_str):
        if not fractional_str:
            return ""
        return f"AND CENTS {self._handle_two_digits(fractional_str)} "

    def _assemble_final_result(self, main_words, fractional_words):
        if not main_words.strip():
            return "ZERO ONLY"
        return f"{main_words.strip()} {fractional_words}ONLY"
