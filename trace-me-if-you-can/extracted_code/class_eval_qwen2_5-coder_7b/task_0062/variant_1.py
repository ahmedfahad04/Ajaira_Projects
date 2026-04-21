class NLPDataProcessor:

    def create_stop_word_list(self):
        stop_word_list = ['a', 'an', 'the']
        return stop_word_list

    def filter_out_stop_words(self, string_list, stop_word_list):
        filtered_words = []
        for sentence in string_list:
            words = sentence.split()
            filtered_words.append([word for word in words if word not in stop_word_list])
        return filtered_words

    def handle_data(self, string_list):
        stop_words = self.create_stop_word_list()
        processed_words = self.filter_out_stop_words(string_list, stop_words)
        return processed_words
