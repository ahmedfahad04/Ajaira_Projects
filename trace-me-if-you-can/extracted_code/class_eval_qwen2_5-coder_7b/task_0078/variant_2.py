import re

class TextProcessor:
    def split_into_sentences(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences

    def count_sentence_words(self, sentence):
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        words = sentence.split()
        return len(words)

    def find_max_word_count(self, text):
        sentences = self.split_into_sentences(text)
        highest_word_count = 0
        for sentence in sentences:
            word_count = self.count_sentence_words(sentence)
            if word_count > highest_word_count:
                highest_word_count = word_count

        return highest_word_count
