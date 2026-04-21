class WordNumberConverter:
    _NUMBERS = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    _TEENS = ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN",
               "EIGHTEEN",
               "NINETEEN"]
    _TENS = ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
    _SCALES = ["", "THOUSAND", "MILLION", "BILLION"]

    def __init__(self):
        self._number_dict = dict(zip(range(10), self._NUMBERS))
        self._teen_dict = dict(zip(range(10, 20), self._TEENS))
        self._tens_dict = dict(zip(range(10, 100, 10), self._TENS))
        self._scale_dict = dict(zip(range(1, 4), self._SCALES))

    def convert(self, number):
        if number is None:
            return ""
        else:
            return self.convert_string(str(number))

    def convert_string(self, number):
        left, right = (number.split('.') + [''])[:2]
        left_reversed = left[::-1]
        converted_parts = [''] * 5

        if len(left_reversed) % 3 == 1:
            left_reversed += "00"
        elif len(left_reversed) % 3 == 2:
            left_reversed += "0"

        major_part = ""
        for i in range(len(left_reversed) // 3):
            converted_parts[i] = left_reversed[3 * i:3 * i + 3][::-1]
            if converted_parts[i] != "000":
                major_part = self.convert_triple(converted_parts[i]) + " " + self.parse_scale(i) + " " + major_part
            else:
                major_part += self.convert_triple(converted_parts[i])

        cents_part = f" AND CENTS {self.convert_double(right)}" if right else ""
        if not major_part.strip():
            return "ZERO ONLY"
        else:
            return f"{major_part.strip()} {cents_part}ONLY"

    def convert_double(self, segment):
        segment = segment.zfill(2)
        if segment[0] == "0":
            return self._number_dict[int(segment[-1])]
        elif segment[0] == "1":
            return self._teen_dict[int(segment)]
        elif segment[1] == "0":
            return self._tens_dict[int(segment[0]) * 10]
        else:
            return self._tens_dict[int(segment[0]) * 10] + " " + self._number_dict[int(segment[-1])]

    def convert_triple(self, segment):
        if segment[0] == "0":
            return self.convert_double(segment[1:])
        elif segment[1:] == "00":
            return f"{self._number_dict[int(segment[0])]} HUNDRED"
        else:
            return f"{self._number_dict[int(segment[0])]} HUNDRED AND {self.convert_double(segment[1:])}"

    def parse_scale(self, index):
        return self._scale_dict[index]
