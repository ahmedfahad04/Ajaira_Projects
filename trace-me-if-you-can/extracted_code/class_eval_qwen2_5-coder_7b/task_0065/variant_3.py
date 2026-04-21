class NumberWordFormatter:
    _NUMBER = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    _NUMBER_TEEN = ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN",
                      "EIGHTEEN",
                      "NINETEEN"]
    _NUMBER_TEN = ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
    _NUMBER_MORE = ["", "THOUSAND", "MILLION", "BILLION"]
    _NUMBER_SUFFIX = ["k", "w", "", "m", "", "", "b", "", "", "t", "", "", "p", "", "", "e"]

    def format(self, value):
        if value is not None:
            return self.format_string(str(value))
        else:
            return ""

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
                major_section = self.trans_three(formatted_parts[i]) + " " + self.parse_more(i) + " " + major_section
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
            return self._NUMBER[int(segment[-1])]
        elif segment[0] == "1":
            return self._NUMBER_TEEN[int(segment) - 10]
        elif segment[1] == "0":
            return self._NUMBER_TEN[int(segment[0]) * 10]
        else:
            return self._NUMBER_TEN[int(segment[0]) * 10] + " " + self._NUMBER[int(segment[-1])]

    def trans_three(self, segment):
        if segment[0] == "0":
            return self.trans_two(segment[1:])
        elif segment[1:] == "00":
            return f"{self._NUMBER[int(segment[0])]} HUNDRED"
        else:
            return f"{self._NUMBER[int(segment[0])]} HUNDRED AND {self.trans_two(segment[1:])}"

    def parse_more(self, index):
        return self._NUMBER_MORE[index]
