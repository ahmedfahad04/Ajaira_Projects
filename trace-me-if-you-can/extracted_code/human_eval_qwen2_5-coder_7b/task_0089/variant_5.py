def shift_characters(s):
       d = 'abcdefghijklmnopqrstuvwxyz'
       out = []
       for c in s:
           if c in d:
               out.append(d[(d.index(c) + 2*2) % 26])
           else:
               out.append(c)
       return ''.join(out)
