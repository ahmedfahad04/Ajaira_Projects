from itertools import cycle

class EncryptionUtils:
    def __init__(self, key):
        self.key = key

    def caesar_cipher(self, plaintext, shift):
        alphabet_size = 26
        result = []
        
        for char in plaintext:
            if not char.isalpha():
                result.append(char)
                continue
                
            offset = ord('A') if char.isupper() else ord('a')
            shifted_value = (ord(char) - offset + shift) % alphabet_size
            result.append(chr(shifted_value + offset))
        
        return ''.join(result)
    
    def vigenere_cipher(self, plain_text):
        key_generator = cycle(self.key.lower())
        result = []
        
        for char in plain_text:
            if char.isalpha():
                key_char = next(key_generator)
                shift = ord(key_char) - ord('a')
                base = ord('a')
                encrypted_ord = (ord(char.lower()) - base + shift) % 26 + base
                encrypted_char = chr(encrypted_ord)
                result.append(encrypted_char.upper() if char.isupper() else encrypted_char)
            else:
                result.append(char)
        
        return ''.join(result)

    def rail_fence_cipher(self, plain_text, rails):
        text_length = len(plain_text)
        rail_matrix = [[None] * text_length for _ in range(rails)]
        
        # Fill the rail matrix
        rail_num = 0
        direction_down = True
        
        for pos, char in enumerate(plain_text):
            rail_matrix[rail_num][pos] = char
            
            if direction_down:
                rail_num += 1
                if rail_num == rails - 1:
                    direction_down = False
            else:
                rail_num -= 1
                if rail_num == 0:
                    direction_down = True
        
        # Read off the rails
        cipher_text = []
        for rail in rail_matrix:
            for char in rail:
                if char is not None:
                    cipher_text.append(char)
        
        return ''.join(cipher_text)
