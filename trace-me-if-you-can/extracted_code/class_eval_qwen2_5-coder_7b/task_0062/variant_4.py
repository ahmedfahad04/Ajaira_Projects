class NLPTextModifier:

    def define_stop_words(self):
        stop_words = ['a', 'an', 'the']
        return stop_words

    def strip_stop_words(self, text_array, stop_words):
        stripped_texts = []
        for text in text_array:
            words = text.split()
            stripped_words = [word for word in words if word not in stop_words]
            stripped_texts.append(stripped_words)
        return stripped_texts

    def run_processing(self, text_array):
        stop_words = self.define_stop_words()
        processed_text = self.strip_stop_words(text_array, stop_words)
        return processed_text
