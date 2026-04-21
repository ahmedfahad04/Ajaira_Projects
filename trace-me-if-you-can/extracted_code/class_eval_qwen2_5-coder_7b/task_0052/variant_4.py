import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class SentenceLemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        lemmatized_words = [self.lemmatize(word, tag) for word, tag in pos_tag(self.tokenize_sentence(sentence))]
        return lemmatized_words

    def get_pos_tag(self, sentence):
        return [tag for word, tag in pos_tag(self.tokenize_sentence(sentence))]

    def lemmatize(self, word, tag):
        pos_tag = tag[0].upper()
        return self.lemmatizer.lemmatize(word, pos=pos_tag) if pos_tag in ['V', 'J', 'R'] else self.lemmatizer.lemmatize(word)

    def tokenize_sentence(self, sentence):
        return word_tokenize(self.remove_punctuation(sentence))

    def remove_punctuation(self, sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
