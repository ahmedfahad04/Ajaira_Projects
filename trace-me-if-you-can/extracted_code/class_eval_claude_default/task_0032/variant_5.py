class DecryptionUtils:
    def __init__(self, key):
        self.key = key
    
    def caesar_decipher(self, ciphertext, shift):
        alphabet_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        alphabet_lower = 'abcdefghijklmnopqrstuvwxyz'
        
        result = []
        for char in ciphertext:
            if char in alphabet_upper:
                old_index = alphabet_upper.index(char)
                new_index = (old_index - shift) % 26
                result.append(alphabet_upper[new_index])
            elif char in alphabet_lower:
                old_index = alphabet_lower.index(char)
                new_index = (old_index - shift) % 26
                result.append(alphabet_lower[new_index])
            else:
                result.append(char)
        
        return ''.join(result)
    
    def vigenere_decipher(self, ciphertext):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        output = []
        key_counter = 0
        
        for character in ciphertext:
            if character.isalpha():
                key_letter = self.key[key_counter % len(self.key)].lower()
                shift_amount = alphabet.index(key_letter)
                
                char_lower = character.lower()
                original_index = alphabet.index(char_lower)
                decrypted_index = (original_index - shift_amount) % 26
                decrypted_letter = alphabet[decrypted_index]
                
                if character.isupper():
                    output.append(decrypted_letter.upper())
                else:
                    output.append(decrypted_letter)
                
                key_counter += 1
            else:
                output.append(character)
        
        return ''.join(output)
    
    def rail_fence_decipher(self, encrypted_text, rails):
        message_length = len(encrypted_text)
        
        # Calculate zigzag path
        zigzag_path = []
        current_row = 0
        direction_down = True
        
        for position in range(message_length):
            zigzag_path.append(current_row)
            
            if direction_down:
                current_row += 1
                if current_row == rails - 1:
                    direction_down = False
            else:
                current_row -= 1
                if current_row == 0:
                    direction_down = True
        
        # Separate encrypted text by rails
        rail_contents = [[] for _ in range(rails)]
        text_index = 0
        
        for rail_num in range(rails):
            for path_position in range(message_length):
                if zigzag_path[path_position] == rail_num:
                    rail_contents[rail_num].append(encrypted_text[text_index])
                    text_index += 1
        
        # Reconstruct original message
        decrypted_message = [''] * message_length
        rail_pointers = [0] * rails
        
        for position in range(message_length):
            target_rail = zigzag_path[position]
            decrypted_message[position] = rail_contents[target_rail][rail_pointers[target_rail]]
            rail_pointers[target_rail] += 1
        
        return ''.join(decrypted_message)
