class TextNormalizer:

    def build_stop_word_list(self):
        stop_word_list = ['a', 'an', 'the']
        return stop_word_list

    def purge_stop_words(self, string_collection, stop_word_list):
        refined_strings = []
        for string in string_collection:
            words = string.split()
            refined_words = [word for word in words if word not in stop_word_list]
            refined_strings.append(refined_words)
        return refined_strings

    def execute_normalization(self, string_collection):
        stop_words = self.build_stop_word_list()
        normalized_data = self.purge_stop_words(string_collection, stop_words)
        return normalized_data
