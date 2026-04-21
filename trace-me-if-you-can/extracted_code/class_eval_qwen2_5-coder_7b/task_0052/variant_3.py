import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

class SentenceProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        lemmatized_words = [self.apply_lemmatization(word, tag) for word, tag in pos_tag(self.remove_punctuation(sentence).split())]
        return lemmatized_words

    def get_pos_tag(self, sentence):
        return [tag for word, tag in pos_tag(self.remove_punctuation(sentence).split())]

    def apply_lemmatization(self, word, tag):
        pos_tag = self.determine_pos_tag(tag)
        return self.lemmatizer.lemmatize(word, pos=pos_tag) if pos_tag else self.lemmatizer.lemmatize(word)

    def determine_pos_tag(self, tag):
        return {'V': 'v', 'J': 'a', 'R': 'r'}.get(tag[0], None)

    def remove_punctuation(self, sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
