class Decryption:
    def __init__(self, key):
        self.key = key
    
    def caesar_decipher(self, text, shift):
        return ''.join(self.shift_char(c, shift) for c in text)
    
    def vigenere_decipher(self, text):
        return ''.join(self.shift_char(c, self.calculate_shift(i, self.key)) for i, c in enumerate(text))
    
    def rail_fence_decipher(self, text, rails):
        fence = ['' for _ in range(rails)]
        rail, step = 0, 1

        for i in range(len(text)):
            fence[rail] += text[i]
            rail += step

            if rail == 0 or rail == rails - 1:
                step = -step

        plain_text = ''
        rail, step = 0, 1
        for _ in range(len(text)):
            plain_text += fence[rail][0]
            fence[rail] = fence[rail][1:]
            rail += step

            if rail == 0 or rail == rails - 1:
                step = -step

        return plain_text
    
    def shift_char(self, char, shift):
        if not char.isalpha():
            return char
        base = 65 if char.isupper() else 97
        return chr((ord(char) - base - shift) % 26 + base)
    
    def calculate_shift(self, index, key):
        return ord(key[index % len(key)].lower()) - ord('a')
