import re
   def calculate_I_sentence_count(S):
       sentences = re.split(r'[.?!]\s*', S)
       count = 0
       for sentence in sentences:
           if sentence[0:2] == 'I ':
               count += 1
       return count
