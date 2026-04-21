class NLPDataProcessor:
    
    def construct_stop_word_list(self):
        return {'a', 'an', 'the'}

    def remove_stop_words(self, string_list, stop_word_list):
        result = []
        for text in string_list:
            filtered_words = [word for word in text.split() if word not in stop_word_list]
            result.append(filtered_words)
        return result

    def process(self, string_list):
        stop_words = self.construct_stop_word_list()
        return self.remove_stop_words(string_list, stop_words)
