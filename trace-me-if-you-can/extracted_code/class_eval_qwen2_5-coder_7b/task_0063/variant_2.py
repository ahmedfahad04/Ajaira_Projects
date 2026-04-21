from collections import Counter
import re

class NLPDataProcessor2:

    def process_data(self, string_list):
        process = lambda s: [word for word in re.sub(r'[^a-zA-Z\s]', '', s.lower()).split()]
        return list(map(process, string_list))

    def calculate_word_frequency(self, words_list):
        word_frequency = Counter(word for sublist in words_list for word in sublist)
        return dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:5])

    def process(self, string_list):
        return self.calculate_word_frequency(self.process_data(string_list))
