import re

class SentenceHandler:
    def segment_sentences(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences

    def get_word_count(self, sentence):
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        words = sentence.split()
        return len(words)

    def get_max_word_count(self, text):
        sentences = self.segment_sentences(text)
        top_word_count = 0
        for sentence in sentences:
            word_count = self.get_word_count(sentence)
            if word_count > top_word_count:
                top_word_count = word_count

        return top_word_count
