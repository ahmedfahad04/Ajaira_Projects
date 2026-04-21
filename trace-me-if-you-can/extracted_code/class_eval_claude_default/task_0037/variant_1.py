class EncryptionUtils:
    def __init__(self, key):
        self.key = key

    def caesar_cipher(self, plaintext, shift):
        def transform_char(char):
            if not char.isalpha():
                return char
            base = ord('A') if char.isupper() else ord('a')
            return chr((ord(char) - base + shift) % 26 + base)
        
        return ''.join(transform_char(char) for char in plaintext)
    
    def vigenere_cipher(self, plain_text):
        result = []
        alpha_count = 0
        
        for char in plain_text:
            if char.isalpha():
                key_char = self.key[alpha_count % len(self.key)]
                shift_amount = ord(key_char.lower()) - ord('a')
                base_char = ord(char.lower()) - ord('a')
                encrypted_base = (base_char + shift_amount) % 26
                encrypted_char = chr(encrypted_base + ord('a'))
                result.append(encrypted_char.upper() if char.isupper() else encrypted_char)
                alpha_count += 1
            else:
                result.append(char)
        
        return ''.join(result)

    def rail_fence_cipher(self, plain_text, rails):
        if rails <= 1:
            return plain_text
            
        rail_positions = []
        current_rail = 0
        direction = 1
        
        for i in range(len(plain_text)):
            rail_positions.append(current_rail)
            current_rail += direction
            if current_rail == rails - 1 or current_rail == 0:
                direction = -direction
        
        encrypted_chars = [''] * rails
        for i, char in enumerate(plain_text):
            encrypted_chars[rail_positions[i]] += char
            
        return ''.join(encrypted_chars)
