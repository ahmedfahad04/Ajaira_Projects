class Decryption:
    def __init__(self, key):
        self.key = key
        self.cipher_methods = {'caesar': lambda text, shift: ''.join(
            chr(((ord(c) - 65 if c.isupper() else ord(c) - 97) - shift) % 26 + 65 if c.isupper() else ((ord(c) - 65 if c.isupper() else ord(c) - 97) - shift) % 26 + 97) if c.isalpha() else c
            for c in text
        ), 'vigenere': lambda text: ''.join(
            chr(((ord(c.lower()) - 97 - (ord(self.key[i % len(self.key)]) - 97)) % 26 + 97) if c.isalpha() else ord(c))
            for i, c in enumerate(text)
        ), 'rail_fence': lambda text, rails: ''.join(
            chr(((ord(c.lower()) - 97 - (ord(self.key[i % len(self.key)]) - 97)) % 26 + 97) if c.isalpha() else ord(c))
            for i, c in enumerate(text)
        )}
    
    def decipher(self, method, text, shift=None, rails=None):
        if method not in self.cipher_methods:
            raise ValueError("Unsupported decryption method")
        return self.cipher_methods[method](text, shift, rails)
