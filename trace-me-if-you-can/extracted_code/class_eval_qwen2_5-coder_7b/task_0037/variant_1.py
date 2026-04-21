class EncryptionUtility:
    def __init__(self, encryptionKey):
        self.encryptionKey = encryptionKey

    def caesarCipher(self, plainText, shiftValue):
        cipherText = ""
        for character in plainText:
            if character.isalpha():
                asciiOffset = 65 if character.isupper() else 97
                shiftedCharacter = chr((ord(character) - asciiOffset + shiftValue) % 26 + asciiOffset)
                cipherText += shiftedCharacter
            else:
                cipherText += character
        return cipherText
    
    def vigenereCipher(self, message):
        encryptedMessage = ""
        keyIndex = 0
        for character in message:
            if character.isalpha():
                shift = ord(self.encryptionKey[keyIndex % len(self.encryptionKey)].lower()) - ord('a')
                encryptedCharacter = chr((ord(character.lower()) - ord('a') + shift) % 26 + ord('a'))
                encryptedMessage += encryptedCharacter.upper() if character.isupper() else encryptedCharacter
                keyIndex += 1
            else:
                encryptedMessage += character
        return encryptedMessage

    def railFenceCipher(self, text, numberOfRails):
        railFence = [['\n' for _ in range(len(text))] for _ in range(numberOfRails)]
        direction = -1
        row, col = 0, 0

        for character in text:
            if row == 0 or row == numberOfRails-1:
                direction = -direction

            railFence[row][col] = character
            col += 1
            row += direction

        encryptedText = ''
        for i in range(numberOfRails):
            for j in range(len(text)):
                if railFence[i][j] != '\n':
                    encryptedText += railFence[i][j]

        return encryptedText
