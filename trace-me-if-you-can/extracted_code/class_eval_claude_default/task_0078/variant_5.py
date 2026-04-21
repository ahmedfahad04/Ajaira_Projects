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
        max_count = 0
        sentence_iter = iter(sentences)
        
        try:
            while True:
                sentence = next(sentence_iter)
                count = self.count_words(sentence)
                max_count = count if count > max_count else max_count
        except StopIteration:
            pass
            
        return max_count
