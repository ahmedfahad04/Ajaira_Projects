import re

class TextAnalyzer:
    def split_sentences(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences

    def count_words_in_sentence(self, sentence):
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        words = sentence.split()
        return len(words)

    def find_largest_word_count(self, text):
        sentences = self.split_sentences(text)
        largest_word_count = 0
        for sentence in sentences:
            word_count = self.count_words_in_sentence(sentence)
            if word_count > largest_word_count:
                largest_word_count = word_count

        return largest_word_count
