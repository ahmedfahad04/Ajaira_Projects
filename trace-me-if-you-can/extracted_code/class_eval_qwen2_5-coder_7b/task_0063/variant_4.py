from collections import Counter
import re

class NLPDataProcessor4:

    def process_data(self, string_list):
        return [[word for word in re.sub(r'[^a-zA-Z\s]', '', string.lower()).split()] for string in string_list]

    def calculate_word_frequency(self, words_list):
        word_frequency = Counter(word for sublist in words_list for word in sublist)
        sorted_word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        return {word: freq for word, freq in sorted_word_frequency[:5]}

    def process(self, string_list):
        return self.calculate_word_frequency(self.process_data(string_list))
