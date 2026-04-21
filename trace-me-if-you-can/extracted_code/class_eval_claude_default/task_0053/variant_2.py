import re
import string


class LongestWord:

    def __init__(self):
        self.word_list = []

    def add_word(self, word):
        self.word_list.append(word)

    def find_longest_word(self, sentence):
        # Preprocess sentence once
        cleaned = re.sub('[%s]' % re.escape(string.punctuation), '', sentence.lower())
        words = cleaned.split()
        
        # Filter and find max in single pass
        max_length = 0
        result = ""
        
        for word in words:
            if word in self.word_list:
                word_length = len(word)
                if word_length > max_length:
                    max_length = word_length
                    result = word
        
        return result
