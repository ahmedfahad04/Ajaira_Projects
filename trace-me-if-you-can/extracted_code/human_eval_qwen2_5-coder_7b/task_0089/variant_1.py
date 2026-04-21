def shift_characters(s):
       d = 'abcdefghijklmnopqrstuvwxyz'
       shift_dict = {c: d[(i+2*2) % 26] for i, c in enumerate(d)}
       return ''.join(shift_dict.get(c, c) for c in s)
