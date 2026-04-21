from collections import Counter, defaultdict
import re

class NLPDataProcessor2:

    def process_data(self, string_list):
        return [re.sub(r'[^a-zA-Z\s]', '', string.lower()).split() 
                for string in string_list]

    def calculate_word_frequency(self, words_list):
        # Manual frequency counting with defaultdict
        frequency_map = defaultdict(int)
        for word_group in words_list:
            for word in word_group:
                frequency_map[word] += 1
        
        # Convert to list of tuples, sort, and take top 5
        frequency_items = list(frequency_map.items())
        frequency_items.sort(key=lambda item: item[1], reverse=True)
        return dict(frequency_items[:5])

    def process(self, string_list):
        words_list = self.process_data(string_list)
        word_frequency_dict = self.calculate_word_frequency(words_list)
        return word_frequency_dict
