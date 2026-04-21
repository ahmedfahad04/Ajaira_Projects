class EncryptionMethods:
    def __init__(self, secretKey):
        self.secretKey = secretKey

    def caesarEncrypt(self, inputText, shift):
        outputText = ""
        for char in inputText:
            if char.isalpha():
                ascii_base = 65 if char.isupper() else 97
                outputText += chr((ord(char) - ascii_base + shift) % 26 + ascii_base)
            else:
                outputText += char
        return outputText
    
    def vigenereEncrypt(self, message):
        encryptedMessage = ""
        key_index = 0
        for char in message:
            if char.isalpha():
                shift = ord(self.secretKey[key_index % len(self.secretKey)].lower()) - ord('a')
                encryptedChar = chr((ord(char.lower()) - ord('a') + shift) % 26 + ord('a'))
                encryptedMessage += encryptedChar.upper() if char.isupper() else encryptedChar
                key_index += 1
            else:
                encryptedMessage += char
        return encryptedMessage

    def railFenceEncrypt(self, plainText, railCount):
        railFenceGrid = [['\n' for _ in range(len(plainText))] for _ in range(railCount)]
        direction = -1
        row, col = 0, 0

        for char in plainText:
            if row == 0 or row == railCount-1:
                direction = -direction

            railFenceGrid[row][col] = char
            col += 1
            row += direction

        encryptedText = ''
        for rail in range(railCount):
            for position in range(len(plainText)):
                if railFenceGrid[rail][position] != '\n':
                    encryptedText += railFenceGrid[rail][position]

        return encryptedText
