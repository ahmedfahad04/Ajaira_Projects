class NumberWordFormatter:
    def __init__(self):
        self.NUMBER = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
        self.NUMBER_TEEN = ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
        self.NUMBER_TEN = ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
        self.NUMBER_MORE = ["", "THOUSAND", "MILLION", "BILLION"]
        self.NUMBER_SUFFIX = ["k", "w", "", "m", "", "", "b", "", "", "t", "", "", "p", "", "", "e"]

    def format(self, x):
        return self._process_number(x) if x is not None else ""

    def _process_number(self, x):
        number_parts = str(x).split('.')
        integer_portion = number_parts[0]
        decimal_portion = number_parts[1] if len(number_parts) > 1 else ""
        
        # Handle the main integer conversion using recursive approach
        main_result = self._convert_integer_recursively(integer_portion, 0)
        
        # Handle decimal part
        decimal_result = f"AND CENTS {self.trans_two(decimal_portion)} " if decimal_portion else ""
        
        if not main_result.strip():
            return "ZERO ONLY"
        
        return f"{main_result.strip()} {decimal_result}ONLY"

    def _convert_integer_recursively(self, num_str, scale_index):
        if not num_str or int(num_str) == 0:
            return ""
        
        if len(num_str) <= 3:
            # Base case: handle up to 3 digits
            padded = num_str.zfill(3)
            if int(padded) == 0:
                return ""
            
            result = self.trans_three(padded)
            scale_word = self.parse_more(scale_index) if scale_index < len(self.NUMBER_MORE) else ""
            
            return f"{result} {scale_word}".strip() if scale_word else result
        
        # Recursive case: split into groups of 3
        split_point = len(num_str) - 3
        left_part = num_str[:split_point]
        right_part = num_str[split_point:]
        
        left_result = self._convert_integer_recursively(left_part, scale_index + 1)
        right_result = self._convert_integer_recursively(right_part, scale_index)
        
        return f"{left_result} {right_result}".strip()

    def trans_two(self, s):
        s = s.zfill(2)
        if s[0] == "0":
            return self.NUMBER[int(s[-1])]
        elif s[0] == "1":
            return self.NUMBER_TEEN[int(s) - 10]
        elif s[1] == "0":
            return self.NUMBER_TEN[int(s[0]) - 1]
        else:
            return self.NUMBER_TEN[int(s[0]) - 1] + " " + self.NUMBER[int(s[-1])]

    def trans_three(self, s):
        if s[0] == "0":
            return self.trans_two(s[1:])
        elif s[1:] == "00":
            return f"{self.NUMBER[int(s[0])]} HUNDRED"
        else:
            return f"{self.NUMBER[int(s[0])]} HUNDRED AND {self.trans_two(s[1:])}"

    def parse_more(self, i):
        return self.NUMBER_MORE[i]
