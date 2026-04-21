class DecryptionUtils:
    UPPERCASE_A = ord('A')
    LOWERCASE_A = ord('a')
    ALPHABET_SIZE = 26
    
    def __init__(self, key):
        self.key = key
    
    def caesar_decipher(self, ciphertext, shift):
        plaintext = []
        
        for char in ciphertext:
            if char.isalpha():
                base_ascii = self.UPPERCASE_A if char.isupper() else self.LOWERCASE_A
                char_position = ord(char) - base_ascii
                new_position = (char_position - shift) % self.ALPHABET_SIZE
                plaintext.append(chr(new_position + base_ascii))
            else:
                plaintext.append(char)
        
        return ''.join(plaintext)
    
    def vigenere_decipher(self, ciphertext):
        result = []
        alphabetic_index = 0
        
        for char in ciphertext:
            if not char.isalpha():
                result.append(char)
                continue
            
            key_char = self.key[alphabetic_index % len(self.key)]
            key_shift = ord(key_char.lower()) - self.LOWERCASE_A
            
            char_lower = char.lower()
            char_position = ord(char_lower) - self.LOWERCASE_A
            decrypted_position = (char_position - key_shift) % self.ALPHABET_SIZE
            decrypted_char = chr(decrypted_position + self.LOWERCASE_A)
            
            final_char = decrypted_char.upper() if char.isupper() else decrypted_char
            result.append(final_char)
            alphabetic_index += 1
        
        return ''.join(result)
    
    def rail_fence_decipher(self, encrypted_text, rails):
        length = len(encrypted_text)
        grid = {}
        
        # Mark positions in zigzag pattern
        row, step = 0, 1
        for col in range(length):
            if (row, col) not in grid:
                grid[(row, col)] = None
            
            row += step
            if row == rails - 1 or row == 0:
                step = -step
        
        # Fill marked positions with encrypted characters
        char_iterator = iter(encrypted_text)
        for r in range(rails):
            for c in range(length):
                if (r, c) in grid:
                    grid[(r, c)] = next(char_iterator)
        
        # Read in zigzag pattern to get plaintext
        plaintext = []
        row, step = 0, 1
        for col in range(length):
            plaintext.append(grid[(row, col)])
            row += step
            if row == rails - 1 or row == 0:
                step = -step
        
        return ''.join(plaintext)
