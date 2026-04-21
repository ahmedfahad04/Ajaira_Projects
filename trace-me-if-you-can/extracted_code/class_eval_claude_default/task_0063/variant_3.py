from collections import Counter
import re

class NLPDataProcessor2:

    def _clean_and_tokenize(self, text):
        """Helper method to clean and tokenize a single string"""
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return cleaned.split()

    def process_data(self, string_list):
        return list(map(self._clean_and_tokenize, string_list))

    def calculate_word_frequency(self, words_list):
        # Aggregate all words first, then count
        all_words = []
        for word_group in words_list:
            all_words.extend(word_group)
        
        counter = Counter(all_words)
        # Use heapq-like approach by getting most common directly
        most_frequent = counter.most_common()
        return {word: count for word, count in most_frequent[:5]}

    def process(self, string_list):
        words_list = self.process_data(string_list)
        word_frequency_dict = self.calculate_word_frequency(words_list)
        return word_frequency_dict
