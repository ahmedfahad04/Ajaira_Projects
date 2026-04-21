from collections import Counter
import re

class NLPDataProcessor2:

    def process_data(self, string_list):
        words_list = []
        pattern = re.compile(r'[^a-zA-Z\s]')  # Compile regex once for efficiency
        
        for string in string_list:
            processed_string = pattern.sub('', string.lower())
            words = processed_string.split()
            words_list.append(words)
        return words_list

    def calculate_word_frequency(self, words_list):
        # Iterative approach building frequency incrementally
        word_frequency = {}
        
        for words in words_list:
            for word in words:
                word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Sort by frequency descending, then slice
        sorted_items = sorted(word_frequency.items(), 
                            key=lambda kv: kv[1], 
                            reverse=True)
        return dict(sorted_items[:5])

    def process(self, string_list):
        words_list = self.process_data(string_list)
        word_frequency_dict = self.calculate_word_frequency(words_list)
        return word_frequency_dict
