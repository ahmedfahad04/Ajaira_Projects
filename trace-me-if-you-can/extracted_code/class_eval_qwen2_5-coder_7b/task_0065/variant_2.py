class NumberToWordAdapter:
    number_dict = {0: "", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE"}
    teen_dict = {10: "TEN", 11: "ELEVEN", 12: "TWELVE", 13: "THIRTEEN", 14: "FOURTEEN", 15: "FIFTEEN", 16: "SIXTEEN", 17: "SEVENTEEN",
                  18: "EIGHTEEN",
                  19: "NINETEEN"}
    tens_dict = {10: "TEN", 20: "TWENTY", 30: "THIRTY", 40: "FORTY", 50: "FIFTY", 60: "SIXTY", 70: "SEVENTY", 80: "EIGHTY", 90: "NINETY"}
    scale_dict = {0: "", 1: "THOUSAND", 2: "MILLION", 3: "BILLION"}

    @staticmethod
    def format_number(num):
        if num is None:
            return ""
        else:
            return NumberToWordAdapter.format_string(str(num))

    @staticmethod
    def format_string(num):
        left, right = (num.split('.') + [''])[:2]
        left_reversed = left[::-1]
        result_parts = [''] * 5

        if len(left_reversed) % 3 == 1:
            left_reversed += "00"
        elif len(left_reversed) % 3 == 2:
            left_reversed += "0"

        major_part = ""
        for i in range(len(left_reversed) // 3):
            result_parts[i] = left_reversed[3 * i:3 * i + 3][::-1]
            if result_parts[i] != "000":
                major_part = NumberToWordAdapter.convert_triple(result_parts[i]) + " " + NumberToWordAdapter.scale_dict[i] + " " + major_part
            else:
                major_part += NumberToWordAdapter.convert_triple(result_parts[i])

        cents_part = f" AND CENTS {NumberToWordAdapter.convert_double(right)}" if right else ""
        if not major_part.strip():
            return "ZERO ONLY"
        else:
            return f"{major_part.strip()} {cents_part}ONLY"

    @staticmethod
    def convert_double(num):
        num = num.zfill(2)
        if num[0] == "0":
            return NumberToWordAdapter.number_dict[int(num[-1])]
        elif num[0] == "1":
            return NumberToWordAdapter.teen_dict[int(num) - 10]
        elif num[1] == "0":
            return NumberToWordAdapter.tens_dict[int(num[0]) * 10]
        else:
            return NumberToWordAdapter.tens_dict[int(num[0]) * 10] + " " + NumberToWordAdapter.number_dict[int(num[-1])]

    @staticmethod
    def convert_triple(num):
        if num[0] == "0":
            return NumberToWordAdapter.convert_double(num[1:])
        elif num[1:] == "00":
            return f"{NumberToWordAdapter.number_dict[int(num[0])]} HUNDRED"
        else:
            return f"{NumberToWordAdapter.number_dict[int(num[0])]} HUNDRED AND {NumberToWordAdapter.convert_double(num[1:])}"
