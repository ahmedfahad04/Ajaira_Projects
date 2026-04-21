import re

class SentenceSplitter:
    def break_sentences(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences

    def extract_words(self, sentence):
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        words = sentence.split()
        return len(words)

    def analyze_text(self, text):
        sentences = self.break_sentences(text)
        max_word_count = 0
        for sentence in sentences:
            word_count = self.extract_words(sentence)
            if word_count > max_word_count:
                max_word_count = word_count

        return max_word_count
