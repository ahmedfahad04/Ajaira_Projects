class LanguageDataHandler:

    def generate_stop_word_set(self):
        stop_word_set = {'a', 'an', 'the'}
        return stop_word_set

    def eliminate_stop_words(self, phrase_list, stop_word_set):
        processed_phrase_list = []
        for phrase in phrase_list:
            words = phrase.split()
            processed_phrase_list.append([word for word in words if word not in stop_word_set])
        return processed_phrase_list

    def perform_processing(self, phrase_list):
        stop_words = self.generate_stop_word_set()
        filtered_phrases = self.eliminate_stop_words(phrase_list, stop_words)
        return filtered_phrases
