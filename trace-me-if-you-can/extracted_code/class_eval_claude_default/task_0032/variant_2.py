class DecryptionUtils:
    def __init__(self, key):
        self.key = key
    
    def _shift_char(self, char, shift, reverse=True):
        if not char.isalpha():
            return char
        
        base = ord('A') if char.isupper() else ord('a')
        offset = ord(char) - base
        new_offset = (offset - shift if reverse else offset + shift) % 26
        return chr(new_offset + base)
    
    def caesar_decipher(self, ciphertext, shift):
        return ''.join(self._shift_char(c, shift) for c in ciphertext)
    
    def vigenere_decipher(self, ciphertext):
        decrypted = []
        key_pos = 0
        
        for char in ciphertext:
            if char.isalpha():
                key_shift = ord(self.key[key_pos % len(self.key)].lower()) - ord('a')
                decrypted.append(self._shift_char(char, key_shift))
                key_pos += 1
            else:
                decrypted.append(char)
        
        return ''.join(decrypted)
    
    def rail_fence_decipher(self, encrypted_text, rails):
        text_len = len(encrypted_text)
        
        # Generate rail positions for each character
        positions = []
        current_rail = 0
        going_down = True
        
        for i in range(text_len):
            positions.append(current_rail)
            
            if current_rail == 0:
                going_down = True
            elif current_rail == rails - 1:
                going_down = False
            
            current_rail += 1 if going_down else -1
        
        # Distribute characters to rails
        rail_chars = {i: [] for i in range(rails)}
        char_index = 0
        
        for rail in range(rails):
            for pos in range(text_len):
                if positions[pos] == rail:
                    rail_chars[rail].append(encrypted_text[char_index])
                    char_index += 1
        
        # Reconstruct message
        rail_indices = {i: 0 for i in range(rails)}
        result = []
        
        for pos in range(text_len):
            rail = positions[pos]
            result.append(rail_chars[rail][rail_indices[rail]])
            rail_indices[rail] += 1
        
        return ''.join(result)
