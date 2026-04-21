from enum import Enum

class Decryption:
    class Cipher(Enum):
        CAESAR = 'caesar'
        VIGENERE = 'vigenere'
        RAIL_FENCE = 'rail_fence'
    
    def __init__(self, key):
        self.key = key
    
    def decipher(self, method, text, shift=None, rails=None):
        if method not in self.Cipher:
            raise ValueError("Unsupported decryption method")
        if method == self.Cipher.CAESAR:
            return self.caesar_decipher(text, shift)
        elif method == self.Cipher.VIGENERE:
            return self.vigenere_decipher(text)
        elif method == self.Cipher.RAIL_FENCE:
            return self.rail_fence_decipher(text, rails)
    
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
        for _ in range(length):
            plain_text += fence[rail][0]
            fence[rail] = fence[rail][1:]
            rail += step

            if rail == 0 or rail == rails - 1:
                step = -step

        return plain_text
