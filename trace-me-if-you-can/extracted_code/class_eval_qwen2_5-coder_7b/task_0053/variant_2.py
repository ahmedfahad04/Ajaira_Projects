import re
import string

class LongWordFinder:

    def __init__(self):
        self.dictionary = {}

    def include_word(self, word):
        self.dictionary[word] = True

    def extract_longest_word(self, text):
        longest_word = ""
        text = text.lower()
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        words = text.split()
        for word in words:
            if word in self.dictionary and len(word) > len(longest_word):
                longest_word = word
        return longest_word
