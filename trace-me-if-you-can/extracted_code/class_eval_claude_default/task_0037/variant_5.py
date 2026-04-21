class EncryptionUtils:
    def __init__(self, key):
        self.key = key
        self._alphabet_lower = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self._alphabet_upper = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    def caesar_cipher(self, plaintext, shift):
        def shift_character(c, shift_amount):
            if c in self._alphabet_lower:
                idx = self._alphabet_lower.index(c)
                return self._alphabet_lower[(idx + shift_amount) % 26]
            elif c in self._alphabet_upper:
                idx = self._alphabet_upper.index(c)
                return self._alphabet_upper[(idx + shift_amount) % 26]
            return c
        
        return ''.join(shift_character(char, shift) for char in plaintext)
    
    def vigenere_cipher(self, plain_text):
        def encrypt_char(plain_char, key_char):
            if not plain_char.isalpha():
                return plain_char
                
            key_shift = self._alphabet_lower.index(key_char.lower())
            
            if plain_char.isupper():
                plain_idx = self._alphabet_upper.index(plain_char)
                return self._alphabet_upper[(plain_idx + key_shift) % 26]
            else:
                plain_idx = self._alphabet_lower.index(plain_char)
                return self._alphabet_lower[(plain_idx + key_shift) % 26]
        
        result = []
        key_index = 0
        
        for char in plain_text:
            if char.isalpha():
                encrypted = encrypt_char(char, self.key[key_index % len(self.key)])
                result.append(encrypted)
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)

    def rail_fence_cipher(self, plain_text, rails):
        def generate_rail_pattern(length, num_rails):
            pattern = []
            rail = 0
            direction = 1
            
            for _ in range(length):
                pattern.append(rail)
                rail += direction
                if rail == num_rails - 1 or rail == 0:
                    direction *= -1
            
            return pattern
        
        pattern = generate_rail_pattern(len(plain_text), rails)
        rail_buckets = [[] for _ in range(rails)]
        
        for char, rail_idx in zip(plain_text, pattern):
            rail_buckets[rail_idx].append(char)
        
        return ''.join(''.join(bucket) for bucket in rail_buckets)
