import re
   def count_sentences_starting_with_I(S):
       sentences = re.split(r'[.?!]\s*', S)
       count = 0
       for sentence in sentences:
           if sentence.startswith('I '):
               count += 1
       return count
