class EntityResolver:
    def __init__(self):
        pass

    def resolve_entities(self, input_string):
        output_list = []
        position = 0
        str_len = len(input_string)

        while position < str_len - 2:
            if input_string[position] == '&' and input_string[position + 1] == '#':
                start_pos = position + 2
                is_hex = False
                first_char = input_string[start_pos]

                if first_char == 'x' or first_char == 'X':
                    start_pos += 1
                    is_hex = True

                if start_pos == str_len:
                    return ''.join(output_list)

                end_pos = start_pos
                while end_pos < str_len and self.is_hex_char(input_string[end_pos]):
                    end_pos += 1

                if end_pos < str_len and input_string[end_pos] == ';':
                    try:
                        value = int(input_string[start_pos:end_pos], 16 if is_hex else 10)
                    except:
                        return ''.join(output_list)

                    output_list.append(chr(value))
                    position = end_pos + 1
                    continue

            output_list.append(input_string[position])
            position += 1

        return ''.join(output_list)

    @staticmethod
    def is_hex_char(char):
        return char.isdigit() or ('a' <= char.lower() <= 'f')
