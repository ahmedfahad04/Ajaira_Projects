import re
import string

class LongestWordStore:

    def __init__(self):
        self.word_list = []

    def include_word(self, word):
        self.word_list.append(word)

    def determine_longest_word(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
        words = re.split(' ', sentence)
        longest_word = ""
        for word in words:
            if word in self.word_list and len(word) > len(longest_word):
                longest_word = word
        return longest_word
