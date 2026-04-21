def caesar_cipher(s):
    result = []
    for char in s:
        if char.islower():
            shifted = chr((ord(char) - ord('a') + 4) % 26 + ord('a'))
            result.append(shifted)
        else:
            result.append(char)
    return ''.join(result)
