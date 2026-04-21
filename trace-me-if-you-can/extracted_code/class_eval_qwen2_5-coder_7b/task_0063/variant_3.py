from collections import Counter
import re

class NLPDataProcessor3:

    def process(self, string_list):
        words_list = [[word for word in re.sub(r'[^a-zA-Z\s]', '', string.lower()).split()] for string in string_list]
        word_frequency = Counter(word for sublist in words_list for word in sublist)
        return dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:5])
