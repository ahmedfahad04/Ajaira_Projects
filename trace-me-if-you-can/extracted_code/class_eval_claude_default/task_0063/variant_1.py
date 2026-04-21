from collections import Counter
import re

class NLPDataProcessor2:
    
    def process(self, string_list):
        # Functional approach using generator expressions and method chaining
        all_words = (
            word 
            for string in string_list 
            for word in re.sub(r'[^a-zA-Z\s]', '', string.lower()).split()
        )
        word_counts = Counter(all_words)
        return dict(word_counts.most_common(5))
    
    def process_data(self, string_list):
        words_list = []
        for string in string_list:
            processed_string = re.sub(r'[^a-zA-Z\s]', '', string.lower())
            words = processed_string.split()
            words_list.append(words)
        return words_list

    def calculate_word_frequency(self, words_list):
        word_frequency = Counter()
        for words in words_list:
            word_frequency.update(words)
        sorted_word_frequency = dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))
        top_5_word_frequency = dict(list(sorted_word_frequency.items())[:5])
        return top_5_word_frequency
