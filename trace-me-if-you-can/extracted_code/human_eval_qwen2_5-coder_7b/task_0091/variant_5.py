import re
   def compute_I_sentence_occurrences(S):
       sentences = re.split(r'[.?!]\s*', S)
       count = 0
       for sentence in sentences:
           if sentence.startswith('I '):
               count += 1
       return count
