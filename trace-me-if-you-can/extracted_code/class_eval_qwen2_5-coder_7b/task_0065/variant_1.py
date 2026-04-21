class DigitToWordConverter:
    def __init__(self):
        self.numbers = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
        self.teens = ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN",
                      "EIGHTEEN",
                      "NINETEEN"]
        self.tens = ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
        self.scale = ["", "THOUSAND", "MILLION", "BILLION"]
        self.suffixes = ["k", "w", "", "m", "", "", "b", "", "", "t", "", "", "p", "", "", "e"]

    def convert(self, num):
        if num is not None:
            return self.convert_string(str(num))
        else:
            return ""

    def convert_string(self, num):
        left, right = (num.split('.') + [''])[:2]
        left_reversed = left[::-1]
        parts = [''] * 5

        if len(left_reversed) % 3 == 1:
            left_reversed += "00"
        elif len(left_reversed) % 3 == 2:
            left_reversed += "0"

        major = ""
        for i in range(len(left_reversed) // 3):
            parts[i] = left_reversed[3 * i:3 * i + 3][::-1]
            if parts[i] != "000":
                major = self.convert_triple(parts[i]) + " " + self.scale[i] + " " + major
            else:
                major += self.convert_triple(parts[i])

        cents = f" AND CENTS {self.convert_double(right)}" if right else ""
        if not major.strip():
            return "ZERO ONLY"
        else:
            return f"{major.strip()} {cents}ONLY"

    def convert_double(self, num):
        num = num.zfill(2)
        if num[0] == "0":
            return self.numbers[int(num[-1])]
        elif num[0] == "1":
            return self.teens[int(num) - 10]
        elif num[1] == "0":
            return self.tens[int(num[0]) - 1]
        else:
            return self.tens[int(num[0]) - 1] + " " + self.numbers[int(num[-1])]

    def convert_triple(self, num):
        if num[0] == "0":
            return self.convert_double(num[1:])
        elif num[1:] == "00":
            return f"{self.numbers[int(num[0])]} HUNDRED"
        else:
            return f"{self.numbers[int(num[0])]} HUNDRED AND {self.convert_double(num[1:])}"

    def parse_scale(self, index):
        return self.scale[index]
