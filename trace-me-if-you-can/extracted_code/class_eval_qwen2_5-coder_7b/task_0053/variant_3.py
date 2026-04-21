import re
import string

class LongestWordHandler:

    def __init__(self):
        self.words = set()

    def append_word(self, word):
        self.words.add(word)

    def identify_longest_word(self, sentence):
        longest_word = ""
        sentence = sentence.lower()
        sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
        words = sentence.split()
        for word in words:
            if word in self.words and len(word) > len(longest_word):
                longest_word = word
        return longest_word
