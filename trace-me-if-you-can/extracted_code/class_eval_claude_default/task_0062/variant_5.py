class NLPDataProcessor:

    def construct_stop_word_list(self):
        stop_word_list = ['a', 'an', 'the']
        return stop_word_list

    def remove_stop_words(self, string_list, stop_word_list):
        def process_single_string(input_string):
            words = input_string.split()
            remaining_words = []
            
            for word in words:
                should_keep = True
                for stop_word in stop_word_list:
                    if word == stop_word:
                        should_keep = False
                        break
                
                if should_keep:
                    remaining_words.append(word)
            
            return remaining_words
        
        return [process_single_string(s) for s in string_list]

    def process(self, string_list):
        stop_word_list = self.construct_stop_word_list()
        words_list = self.remove_stop_words(string_list, stop_word_list)
        return words_list
