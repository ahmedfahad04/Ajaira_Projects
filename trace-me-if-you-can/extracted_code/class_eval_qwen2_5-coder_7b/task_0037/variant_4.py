class EncryptionTool:
    def __init__(self, key):
        self.key = key

    def caesarShift(self, text, shift):
        cipher = ""
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                shifted_char = chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
                cipher += shifted_char
            else:
                cipher += char
        return cipher
    
    def vigenereEncrypt(self, message):
        encrypted_message = ""
        key_index = 0
        for char in message:
            if char.isalpha():
                shift = ord(self.key[key_index % len(self.key)].lower()) - ord('a')
                encrypted_char = chr((ord(char.lower()) - ord('a') + shift) % 26 + ord('a'))
                encrypted_message += encrypted_char.upper() if char.isupper() else encrypted_char
                key_index += 1
            else:
                encrypted_message += char
        return encrypted_message

    def railFenceEncrypt(self, text, rails):
        rail_fence = [['\n' for _ in range(len(text))] for _ in range(rails)]
        direction = -1
        row, col = 0, 0

        for char in text:
            if row == 0 or row == rails-1:
                direction = -direction

            rail_fence[row][col] = char
            col += 1
            row += direction

        encrypted_text = ''
        for rail in range(rails):
            for position in range(len(text)):
                if rail_fence[rail][position] != '\n':
                    encrypted_text += rail_fence[rail][position]

        return encrypted_text
