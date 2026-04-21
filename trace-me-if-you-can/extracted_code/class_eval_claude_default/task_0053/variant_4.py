import re
import string


class LongestWord:

    def __init__(self):
        self.word_list = []

    def add_word(self, word):
        self.word_list.append(word)

    def find_longest_word(self, sentence):
        # Create translation table for punctuation removal
        translator = str.maketrans('', '', string.punctuation)
        clean_sentence = sentence.lower().translate(translator)
        
        # Use generator expression with max
        valid_words = (word for word in clean_sentence.split() if word in self.word_list)
        
        try:
            return max(valid_words, key=len)
        except ValueError:  # empty sequence
            return ""
