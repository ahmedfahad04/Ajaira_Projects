class NumericEntityReplacer:
    def __init__(self):
        pass

    def replace(self, input_str):
        result = []
        index = 0
        str_len = len(input_str)

        while index < str_len - 2:
            if input_str[index] == '&' and input_str[index + 1] == '#':
                start_index = index + 2
                is_hex = False
                first_char = input_str[start_index]

                if first_char == 'x' or first_char == 'X':
                    start_index += 1
                    is_hex = True

                if start_index == str_len:
                    return ''.join(result)

                end_index = start_index
                while end_index < str_len and self.is_hex_char(input_str[end_index]):
                    end_index += 1

                if end_index < str_len and input_str[end_index] == ';':
                    try:
                        value = int(input_str[start_index:end_index], 16 if is_hex else 10)
                    except:
                        return ''.join(result)

                    result.append(chr(value))
                    index = end_index + 1
                    continue

            result.append(input_str[index])
            index += 1

        return ''.join(result)

    @staticmethod
    def is_hex_char(char):
        return char.isdigit() or ('a' <= char.lower() <= 'f')
