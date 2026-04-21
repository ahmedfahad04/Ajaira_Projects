class Decryption:
    def __init__(self, key):
        self.key = key
    
    def caesar_decipher(self, text, shift):
        return ''.join(
            chr(((ord(c) - 65 if c.isupper() else ord(c) - 97) - shift) % 26 + 65 if c.isupper() else ((ord(c) - 65 if c.isupper() else ord(c) - 97) - shift) % 26 + 97) if c.isalpha() else c
            for c in text
        )
    
    def vigenere_decipher(self, text):
        return ''.join(
            chr(((ord(c.lower()) - 97 - (ord(self.key[i % len(self.key)]) - 97)) % 26 + 97) if c.isalpha() else ord(c))
            for i, c in enumerate(text)
        )
    
    def rail_fence_decipher(self, text, rails):
        length = len(text)
        fence = ['' for _ in range(rails)]
        rail, step = 0, 1

        for i in range(length):
            fence[rail] += text[i]
            rail += step

            if rail == 0 or rail == rails - 1:
                step = -step

        plain_text = ''
        rail, step = 0, 1
        for i in range(length):
            plain_text += fence[rail][0]
            fence[rail] = fence[rail][1:]
            rail += step

            if rail == 0 or rail == rails - 1:
                step = -step

        return plain_text
