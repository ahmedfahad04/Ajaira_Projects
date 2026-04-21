import re

class SentenceAnalyzer:
    def tokenize_sentences(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences

    def word_count_in_sentence(self, sentence):
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        words = sentence.split()
        return len(words)

    def determine_highest_word_count(self, text):
        sentences = self.tokenize_sentences(text)
        max_word_num = 0
        for sentence in sentences:
            word_num = self.word_count_in_sentence(sentence)
            if word_num > max_word_num:
                max_word_num = word_num

        return max_word_num
