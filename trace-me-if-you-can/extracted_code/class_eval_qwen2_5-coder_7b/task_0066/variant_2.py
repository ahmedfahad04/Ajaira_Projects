class NumericEntityReplacer:
    def __init__(self):
        pass

    def replace_entities(self, input_str):
        output = []
        current_pos = 0
        str_len = len(input_str)

        while current_pos < str_len - 2:
            if input_str[current_pos] == '&' and input_str[current_pos + 1] == '#':
                start_pos = current_pos + 2
                is_hex = False
                first_char = input_str[start_pos]

                if first_char == 'x' or first_char == 'X':
                    start_pos += 1
                    is_hex = True

                if start_pos == str_len:
                    return ''.join(output)

                end_pos = start_pos
                while end_pos < str_len and self.is_hex_char(input_str[end_pos]):
                    end_pos += 1

                if end_pos < str_len and input_str[end_pos] == ';':
                    try:
                        value = int(input_str[start_pos:end_pos], 16 if is_hex else 10)
                    except:
                        return ''.join(output)

                    output.append(chr(value))
                    current_pos = end_pos + 1
                    continue

            output.append(input_str[current_pos])
            current_pos += 1

        return ''.join(output)

    @staticmethod
    def is_hex_char(char):
        return char.isdigit() or ('a' <= char.lower() <= 'f')
