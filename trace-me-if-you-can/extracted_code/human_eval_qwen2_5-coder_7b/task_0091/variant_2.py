import re
   def tally_sentences_with_I(S):
       sentences = re.split(r'[.?!]\s*', S)
       return len([sentence for sentence in sentences if sentence[:2] == 'I '])
