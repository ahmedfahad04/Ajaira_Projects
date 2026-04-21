def is_lowercase(c):
       return 'a' <= c <= 'z'

   def shift_character(c):
       if is_lowercase(c):
           return chr((ord(c) - ord('a') + 2*2) % 26 + ord('a'))
       return c

   def shift_characters(s):
       return ''.join(shift_character(c) for c in s)
