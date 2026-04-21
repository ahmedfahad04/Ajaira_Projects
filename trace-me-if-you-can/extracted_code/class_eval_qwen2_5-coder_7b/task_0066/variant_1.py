class EntityDecoder:
    def __init__(self):
        pass

    def decode(self, text):
        decoded = []
        index = 0
        text_length = len(text)

        while index < text_length - 2:
            if text[index] == '&' and text[index + 1] == '#':
                start_index = index + 2
                is_hexadecimal = False
                first_char = text[start_index]

                if first_char == 'x' or first_char == 'X':
                    start_index += 1
                    is_hexadecimal = True

                if start_index == text_length:
                    return ''.join(decoded)

                end_index = start_index
                while end_index < text_length and self.is_hexadecimal_char(text[end_index]):
                    end_index += 1

                if end_index < text_length and text[end_index] == ';':
                    try:
                        decoded_value = int(text[start_index:end_index], 16 if is_hexadecimal else 10)
                    except:
                        return ''.join(decoded)

                    decoded.append(chr(decoded_value))
                    index = end_index + 1
                    continue

            decoded.append(text[index])
            index += 1

        return ''.join(decoded)

    @staticmethod
    def is_hexadecimal_char(character):
        return character.isdigit() or ('a' <= character.lower() <= 'f')
