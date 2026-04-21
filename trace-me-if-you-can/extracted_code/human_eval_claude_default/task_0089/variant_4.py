def caesar_cipher(s):
    alphabet_map = {chr(i): chr((i - ord('a') + 4) % 26 + ord('a')) 
                    for i in range(ord('a'), ord('z') + 1)}
    
    return ''.join(alphabet_map.get(char, char) for char in s)
