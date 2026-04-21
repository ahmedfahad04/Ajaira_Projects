from itertools import cycle

class DecryptionUtils:
    def __init__(self, key):
        self.key = key
    
    def caesar_decipher(self, ciphertext, shift):
        def decrypt_char(c):
            if not c.isalpha():
                return c
            base = 65 if c.isupper() else 97
            return chr((ord(c) - base - shift) % 26 + base)
        
        return ''.join(map(decrypt_char, ciphertext))
    
    def vigenere_decipher(self, ciphertext):
        key_cycle = cycle(self.key.lower())
        result = []
        
        for char in ciphertext:
            if char.isalpha():
                key_char = next(key_cycle)
                shift_val = ord(key_char) - ord('a')
                
                decrypted_ord = (ord(char.lower()) - ord('a') - shift_val) % 26 + ord('a')
                decrypted_char = chr(decrypted_ord)
                
                result.append(decrypted_char.upper() if char.isupper() else decrypted_char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    def rail_fence_decipher(self, encrypted_text, rails):
        n = len(encrypted_text)
        
        # Create rail assignment map
        rail_map = []
        rail, direction = 0, 1
        
        for _ in range(n):
            rail_map.append(rail)
            rail += direction
            if rail in (0, rails - 1):
                direction *= -1
        
        # Group characters by rail
        rails_content = [[] for _ in range(rails)]
        encrypted_iter = iter(encrypted_text)
        
        for target_rail in range(rails):
            for pos in range(n):
                if rail_map[pos] == target_rail:
                    rails_content[target_rail].append(next(encrypted_iter))
        
        # Reconstruct original text
        rail_iterators = [iter(rail_content) for rail_content in rails_content]
        
        return ''.join(
            next(rail_iterators[rail_map[pos]]) 
            for pos in range(n)
        )
