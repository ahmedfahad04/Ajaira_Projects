class NumericEntityProcessor:
    def __init__(self):
        pass

    def process(self, input_text):
        result = []
        pos = 0
        text_length = len(input_text)

        while pos < text_length - 2:
            if input_text[pos] == '&' and input_text[pos + 1] == '#':
                start = pos + 2
                is_hex = False
                first_char = input_text[start]

                if first_char == 'x' or first_char == 'X':
                    start += 1
                    is_hex = True

                if start == text_length:
                    return ''.join(result)

                end = start
                while end < text_length and self.is_hex_char(input_text[end]):
                    end += 1

                if end < text_length and input_text[end] == ';':
                    try:
                        value = int(input_text[start:end], 16 if is_hex else 10)
                    except:
                        return ''.join(result)

                    result.append(chr(value))
                    pos = end + 1
                    continue

            result.append(input_text[pos])
            pos += 1

        return ''.join(result)

    @staticmethod
    def is_hex_char(char):
        return char.isdigit() or ('a' <= char.lower() <= 'f')
