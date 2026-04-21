import re
import string

class WordRepository:

    def __init__(self):
        self.vocab = {}

    def add_entry(self, word):
        self.vocab[word] = True

    def get_longest_word(self, passage):
        longest_word = ""
        passage = passage.lower()
        passage = re.sub('[%s]' % re.escape(string.punctuation), '', passage)
        words = passage.split()
        for word in words:
            if word in self.vocab and len(word) > len(longest_word):
                longest_word = word
        return longest_word
