import re


class SplitSentence:

    def split_sentences(self, sentences_string):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', sentences_string)
        return sentences

    def count_words(self, sentence):
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        words = sentence.split()
        return len(words)

    def process_text_file(self, sentences_string):
        sentences = self.split_sentences(sentences_string)
        word_counts = [self.count_words(sentence) for sentence in sentences]
        return max(word_counts) if word_counts else 0
