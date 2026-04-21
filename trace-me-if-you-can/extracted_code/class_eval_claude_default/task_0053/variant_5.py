import re
import string


class LongestWord:

    def __init__(self):
        self.word_list = []

    def add_word(self, word):
        self.word_list.append(word)

    def find_longest_word(self, sentence):
        # Dictionary to store word lengths for quick lookup
        word_lengths = {word: len(word) for word in self.word_list}
        
        # Clean sentence
        clean_sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence.lower())
        
        # Track best match
        best_word = ""
        best_length = 0
        
        # Single pass through sentence words
        for word in clean_sentence.split():
            if word in word_lengths and word_lengths[word] > best_length:
                best_word = word
                best_length = word_lengths[word]
        
        return best_word
