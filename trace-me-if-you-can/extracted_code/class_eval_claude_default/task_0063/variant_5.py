from collections import Counter
import re
from functools import reduce
from operator import add

class NLPDataProcessor2:

    def process_data(self, string_list):
        def clean_and_split(s):
            return re.sub(r'[^a-zA-Z\s]', '', s.lower()).split()
        
        words_list = []
        for string in string_list:
            words_list.append(clean_and_split(string))
        return words_list

    def calculate_word_frequency(self, words_list):
        # Functional reduce approach to flatten and count
        if not words_list:
            return {}
            
        flattened_words = reduce(add, words_list, [])
        word_counter = Counter(flattened_words)
        
        # Get top 5 using sorted with custom key
        top_words = sorted(word_counter.keys(), 
                          key=word_counter.get, 
                          reverse=True)[:5]
        
        return {word: word_counter[word] for word in top_words}

    def process(self, string_list):
        words_list = self.process_data(string_list)
        word_frequency_dict = self.calculate_word_frequency(words_list)
        return word_frequency_dict
