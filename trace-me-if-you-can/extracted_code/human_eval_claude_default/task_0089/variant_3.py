def caesar_cipher(s):
    def shift_char(c):
        return chr((ord(c) - ord('a') + 4) % 26 + ord('a')) if 'a' <= c <= 'z' else c
    
    return ''.join(map(shift_char, s))
