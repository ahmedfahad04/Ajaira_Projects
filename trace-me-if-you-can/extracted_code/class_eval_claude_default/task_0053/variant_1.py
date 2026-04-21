import re
import string
from functools import reduce


class LongestWord:

    def __init__(self):
        self.word_list = set()

    def add_word(self, word):
        self.word_list.add(word)

    def find_longest_word(self, sentence):
        cleaned_sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence.lower())
        words = cleaned_sentence.split()
        valid_words = [word for word in words if word in self.word_list]
        return reduce(lambda a, b: a if len(a) > len(b) else b, valid_words, "")
