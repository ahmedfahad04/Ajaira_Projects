class NLPDataProcessor:

    def construct_stop_word_list(self):
        stop_word_list = ['a', 'an', 'the']
        return stop_word_list

    def remove_stop_words(self, string_list, stop_word_list):
        stop_word_set = set(stop_word_list)
        processed_strings = []
        
        for text in string_list:
            words = text.split()
            clean_words = []
            for word in words:
                if word not in stop_word_set:
                    clean_words.append(word)
            processed_strings.append(clean_words)
        
        return processed_strings

    def process(self, string_list):
        stop_word_list = self.construct_stop_word_list()
        words_list = self.remove_stop_words(string_list, stop_word_list)
        return words_list
