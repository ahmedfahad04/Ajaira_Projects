class EncryptionService:
    def __init__(self, key):
        self.key = key

    def caesarEncrypt(self, plaintext, shift):
        ciphertext = ""
        for character in plaintext:
            if character.isalpha():
                ascii_offset = 65 if character.isupper() else 97
                shifted_character = chr((ord(character) - ascii_offset + shift) % 26 + ascii_offset)
                ciphertext += shifted_character
            else:
                ciphertext += character
        return ciphertext
    
    def vigenereEncrypt(self, message):
        encrypted_message = ""
        key_index = 0
        for character in message:
            if character.isalpha():
                shift = ord(self.key[key_index % len(self.key)].lower()) - ord('a')
                encrypted_char = chr((ord(character.lower()) - ord('a') + shift) % 26 + ord('a'))
                encrypted_message += encrypted_char.upper() if character.isupper() else encrypted_char
                key_index += 1
            else:
                encrypted_message += character
        return encrypted_message

    def railFenceEncrypt(self, plain_text, rail_count):
        rail_fence = [['\n' for _ in range(len(plain_text))] for _ in range(rail_count)]
        direction = -1
        row, col = 0, 0

        for character in plain_text:
            if row == 0 or row == rail_count-1:
                direction = -direction

            rail_fence[row][col] = character
            col += 1
            row += direction

        encrypted_text = ''
        for i in range(rail_count):
            for j in range(len(plain_text)):
                if rail_fence[i][j] != '\n':
                    encrypted_text += rail_fence[i][j]

        return encrypted_text
