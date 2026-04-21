class NLPDataProcessor:
    STOP_WORDS = ['a', 'an', 'the']

    def construct_stop_word_list(self):
        return self.STOP_WORDS[:]

    def remove_stop_words(self, string_list, stop_word_list):
        answer = []
        stop_word_lookup = frozenset(stop_word_list)
        
        for string in string_list:
            tokens = string.split()
            filtered_tokens = []
            
            i = 0
            while i < len(tokens):
                if tokens[i] not in stop_word_lookup:
                    filtered_tokens.append(tokens[i])
                i += 1
            
            answer.append(filtered_tokens)
        return answer

    def process(self, string_list):
        stop_word_list = self.construct_stop_word_list()
        words_list = self.remove_stop_words(string_list, stop_word_list)
        return words_list
