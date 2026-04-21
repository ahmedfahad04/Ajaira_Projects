import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')


class Lemmatization:
    POS_CONVERSIONS = {
        'V': 'v',
        'J': 'a',
        'R': 'r'
    }

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        def process_word_tag(word_tag_pair):
            word, tag = word_tag_pair
            wordnet_pos = self._convert_pos_tag(tag)
            return self.lemmatizer.lemmatize(word, pos=wordnet_pos) if wordnet_pos else self.lemmatizer.lemmatize(word)

        clean_text = self.remove_punctuation(sentence)
        word_tag_pairs = pos_tag(word_tokenize(clean_text))
        
        return [process_word_tag(pair) for pair in word_tag_pairs]

    def _convert_pos_tag(self, tag):
        for prefix, wordnet_pos in self.POS_CONVERSIONS.items():
            if tag.startswith(prefix):
                return wordnet_pos
        return None

    def get_pos_tag(self, sentence):
        clean_text = self.remove_punctuation(sentence)
        word_tag_pairs = pos_tag(word_tokenize(clean_text))
        return [pair[1] for pair in word_tag_pairs]

    def remove_punctuation(self, sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
