import re
   def find_I_statements(S):
       sentences = re.split(r'[.?!]\s*', S)
       count = 0
       for sentence in sentences:
           if sentence[0:2] == 'I ':
               count += 1
       return count
