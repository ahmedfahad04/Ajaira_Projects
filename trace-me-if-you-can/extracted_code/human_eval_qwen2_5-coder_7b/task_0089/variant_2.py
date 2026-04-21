def shift_characters(s):
       d = 'abcdefghijklmnopqrstuvwxyz'
       shifted = [d[(d.index(c) + 2*2) % 26] if c in d else c for c in s]
       return ''.join(shifted)
