def caesar_cipher(s):
    import string
    
    lowercase = string.ascii_lowercase
    shifted = lowercase[4:] + lowercase[:4]
    cipher_dict = dict(zip(lowercase, shifted))
    
    output = ''
    for character in s:
        output += cipher_dict.get(character, character)
    return output
