class TextProcessor:

    def init_stop_words(self):
        stop_words = ['a', 'an', 'the']
        return stop_words

    def exclude_stop_words(self, text_list, stop_words):
        cleaned_list = []
        for text in text_list:
            words = text.split()
            cleaned_words = [word for word in words if word not in stop_words]
            cleaned_list.append(cleaned_words)
        return cleaned_list

    def execute_processing(self, text_list):
        stop_words = self.init_stop_words()
        result = self.exclude_stop_words(text_list, stop_words)
        return result
