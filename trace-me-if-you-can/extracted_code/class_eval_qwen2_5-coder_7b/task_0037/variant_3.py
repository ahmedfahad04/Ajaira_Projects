class CryptoOperations:
    def __init__(self, key):
        self.key = key

    def rotateCipher(self, text, shift):
        result = ""
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                rotated_char = chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
                result += rotated_char
            else:
                result += char
        return result
    
    def vigenereEncrypt(self, input_string):
        encrypted_string = ""
        key_index = 0
        for char in input_string:
            if char.isalpha():
                shift = ord(self.key[key_index % len(self.key)].lower()) - ord('a')
                encrypted_char = chr((ord(char.lower()) - ord('a') + shift) % 26 + ord('a'))
                encrypted_string += encrypted_char.upper() if char.isupper() else encrypted_char
                key_index += 1
            else:
                encrypted_string += char
        return encrypted_string

    def railFenceEncrypt(self, text, rail):
        rail_fence = [['\n' for _ in range(len(text))] for _ in range(rail)]
        direction = -1
        row, col = 0, 0

        for char in text:
            if row == 0 or row == rail-1:
                direction = -direction

            rail_fence[row][col] = char
            col += 1
            row += direction

        encrypted_text = ''
        for i in range(rail):
            for j in range(len(text)):
                if rail_fence[i][j] != '\n':
                    encrypted_text += rail_fence[i][j]

        return encrypted_text
