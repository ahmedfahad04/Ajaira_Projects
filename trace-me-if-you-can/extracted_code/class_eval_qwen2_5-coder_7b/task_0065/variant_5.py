class NumberToWordFormatter:
    _NUMBERS = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    _TEENS = ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN",
               "EIGHTEEN",
               "NINETEEN"]
    _TENS = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
    _SCALES = ["", "THOUSAND", "MILLION", "BILLION"]

    def __init__(self):
        self._number_dict = dict(zip(range(10), self._NUMBERS))
        self._teen_dict = dict(zip(range(10, 20), self._TEENS))
        self._tens_dict = dict(zip(range(2, 100, 10), self._TENS))
        self._scale_dict = dict(zip(range(1, 4), self._SCALES))

    def format(self, value):
        if value is None:
            return ""
        else:
            return self.format_string(str(value))

    def format_string(self, value):
        left, right = (value.split('.') + [''])[:2]
        left_reversed = left[::-1]
        formatted_parts = [''] * 5

        if len(left_reversed) % 3 == 1:
            left_reversed += "00"
        elif len(left_reversed) % 3 == 2:
            left_reversed += "0"

        major_section = ""
        for i in range(len(left_reversed) // 3):
            formatted_parts[i] = left_reversed[3 * i:3 * i + 3][::-1]
            if formatted_parts[i] != "000":
                major_section = self.trans_three(formatted_parts[i]) + " " + self.parse_scale(i) + " " + major_section
            else:
                major_section += self.trans_three(formatted_parts[i])

        cents_section = f" AND CENTS {self.trans_two(right)}" if right else ""
        if not major_section.strip():
            return "ZERO ONLY"
        else:
            return f"{major_section.strip()} {cents_section}ONLY"

    def trans_two(self, segment):
        segment = segment.zfill(2)
        if segment[0] == "0":
            return self._number_dict[int(segment[-1])]
        elif segment[0] == "1":
            return self._teen_dict[int(segment) - 10]
        elif segment[1] == "0":
            return self._tens_dict[int(segment[0])]
        else:
            return self._tens_dict[int(segment[0])] + " " + self._number_dict[int(segment[-1])]

    def trans_three(self, segment):
        if segment[0] == "0":
            return self.trans_two(segment[1:])
        elif segment[1:] == "00":
            return f"{self._number_dict[int(segment[0])]} HUNDRED"
        else:
            return f"{self._number_dict[int(segment[0])]} HUNDRED AND {self.trans_two(segment[1:])}"

    def parse_scale(self, index):
        return self._scale_dict[index]
