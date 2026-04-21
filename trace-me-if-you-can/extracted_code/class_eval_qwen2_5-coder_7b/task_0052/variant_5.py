import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

class SentenceNormalizer:
    POS_MAP = {'V': 'v', 'J': 'a', 'R': 'r'}

    @staticmethod
    def lemmatize_sentence(sentence):
        lemmatized_words = [SentenceNormalizer.apply_lemmatization(word, tag) for word, tag in pos_tag(SentenceNormalizer.remove_punctuation(sentence).split())]
        return lemmatized_words

    @staticmethod
    def get_pos_tag(sentence):
        return [tag for word, tag in pos_tag(SentenceNormalizer.remove_punctuation(sentence).split())]

    @staticmethod
    def apply_lemmatization(word, tag):
        pos_tag = SentenceNormalizer.determine_pos_tag(tag)
        return WordNetLemmatizer().lemmatize(word, pos=pos_tag) if pos_tag else WordNetLemmatizer().lemmatize(word)

    @staticmethod
    def determine_pos_tag(tag):
        return SentenceNormalizer.POS_MAP.get(tag[0], None)

    @staticmethod
    def remove_punctuation(sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
