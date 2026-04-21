import re
import string


class LongestWord:

    def __init__(self):
        self.word_list = []

    def add_word(self, word):
        self.word_list.append(word)

    def find_longest_word(self, sentence):
        def clean_and_tokenize(text):
            no_punct = re.sub('[%s]' % re.escape(string.punctuation), '', text.lower())
            return no_punct.split()
        
        def is_valid_word(word):
            return word in self.word_list
        
        tokens = clean_and_tokenize(sentence)
        valid_tokens = filter(is_valid_word, tokens)
        return max(valid_tokens, key=len, default="")
