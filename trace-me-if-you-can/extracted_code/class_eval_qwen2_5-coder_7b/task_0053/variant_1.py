import re
import string

class WordProcessor:

    def __init__(self):
        self.word_set = set()

    def ingest_word(self, word):
        self.word_set.add(word)

    def retrieve_longest_word(self, phrase):
        longest_word = ""
        phrase = phrase.lower()
        phrase = re.sub('[%s]' % re.escape(string.punctuation), '', phrase)
        words = phrase.split()
        for word in words:
            if word in self.word_set and len(word) > len(longest_word):
                longest_word = word
        return longest_word
