def shift_characters(s):
       d = 'abcdefghijklmnopqrstuvwxyz'
       shift = lambda c: d[(d.index(c) + 2*2) % 26] if c in d else c
       return ''.join(shift(c) for c in s)
