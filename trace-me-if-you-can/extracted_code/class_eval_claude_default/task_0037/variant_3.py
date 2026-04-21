class EncryptionUtils:
    def __init__(self, key):
        self.key = key

    def caesar_cipher(self, plaintext, shift):
        ciphertext = []
        for char in plaintext:
            if char.isalpha():
                is_upper = char.isupper()
                char = char.lower()
                shifted = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
                ciphertext.append(shifted.upper() if is_upper else shifted)
            else:
                ciphertext.append(char)
        return ''.join(ciphertext)
    
    def vigenere_cipher(self, plain_text):
        encrypted = []
        key_pos = 0
        
        for char in plain_text:
            if char.isalpha():
                key_shift = ord(self.key[key_pos % len(self.key)].lower()) - ord('a')
                char_lower = char.lower()
                encrypted_lower = chr((ord(char_lower) - ord('a') + key_shift) % 26 + ord('a'))
                encrypted.append(encrypted_lower.upper() if char.isupper() else encrypted_lower)
                key_pos += 1
            else:
                encrypted.append(char)
        
        return ''.join(encrypted)

    def rail_fence_cipher(self, plain_text, rails):
        if rails == 1:
            return plain_text
            
        rails_content = {i: [] for i in range(rails)}
        current_rail = 0
        going_down = True
        
        for char in plain_text:
            rails_content[current_rail].append(char)
            
            if going_down:
                current_rail += 1
                if current_rail == rails - 1:
                    going_down = False
            else:
                current_rail -= 1
                if current_rail == 0:
                    going_down = True
        
        result = []
        for rail_num in range(rails):
            result.extend(rails_content[rail_num])
        
        return ''.join(result)
