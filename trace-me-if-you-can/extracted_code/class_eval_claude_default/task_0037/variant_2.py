import string

class EncryptionUtils:
    def __init__(self, key):
        self.key = key
        self.uppercase = string.ascii_uppercase
        self.lowercase = string.ascii_lowercase

    def caesar_cipher(self, plaintext, shift):
        translation_table = str.maketrans(
            self.uppercase + self.lowercase,
            self.uppercase[shift:] + self.uppercase[:shift] + 
            self.lowercase[shift:] + self.lowercase[:shift]
        )
        return plaintext.translate(translation_table)
    
    def vigenere_cipher(self, plain_text):
        encrypted_text = ""
        key_cycle = 0
        
        for character in plain_text:
            if character in self.uppercase or character in self.lowercase:
                key_letter = self.key[key_cycle % len(self.key)].lower()
                shift_value = self.lowercase.index(key_letter)
                
                if character in self.uppercase:
                    char_index = self.uppercase.index(character)
                    new_char = self.uppercase[(char_index + shift_value) % 26]
                else:
                    char_index = self.lowercase.index(character)
                    new_char = self.lowercase[(char_index + shift_value) % 26]
                
                encrypted_text += new_char
                key_cycle += 1
            else:
                encrypted_text += character
                
        return encrypted_text

    def rail_fence_cipher(self, plain_text, rails):
        rail_groups = [[] for _ in range(rails)]
        rail_index = 0
        step = 1
        
        for char in plain_text:
            rail_groups[rail_index].append(char)
            rail_index += step
            if rail_index == rails - 1 or rail_index == 0:
                step *= -1
        
        return ''.join(''.join(rail) for rail in rail_groups)
