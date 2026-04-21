class DecryptionUtils:
    def __init__(self, key):
        self.key = key
    
    def caesar_decipher(self, ciphertext, shift):
        return ''.join(
            chr((ord(char) - (65 if char.isupper() else 97) - shift) % 26 + (65 if char.isupper() else 97))
            if char.isalpha() else char
            for char in ciphertext
        )
    
    def vigenere_decipher(self, ciphertext):
        result = []
        alpha_count = 0
        
        for char in ciphertext:
            if char.isalpha():
                key_char = self.key[alpha_count % len(self.key)]
                shift = ord(key_char.lower()) - ord('a')
                decrypted = chr((ord(char.lower()) - ord('a') - shift) % 26 + ord('a'))
                result.append(decrypted.upper() if char.isupper() else decrypted)
                alpha_count += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    def rail_fence_decipher(self, encrypted_text, rails):
        if rails == 1:
            return encrypted_text
            
        # Create zigzag pattern markers
        pattern = []
        row = 0
        direction = 1
        
        for i in range(len(encrypted_text)):
            pattern.append(row)
            row += direction
            if row == rails - 1 or row == 0:
                direction = -direction
        
        # Fill the fence with characters
        fence = [[] for _ in range(rails)]
        char_idx = 0
        
        for rail in range(rails):
            for pos in range(len(encrypted_text)):
                if pattern[pos] == rail:
                    fence[rail].append(encrypted_text[char_idx])
                    char_idx += 1
                else:
                    fence[rail].append(None)
        
        # Read the message
        result = []
        for pos in range(len(encrypted_text)):
            rail = pattern[pos]
            result.append(fence[rail][pos])
        
        return ''.join(result)
