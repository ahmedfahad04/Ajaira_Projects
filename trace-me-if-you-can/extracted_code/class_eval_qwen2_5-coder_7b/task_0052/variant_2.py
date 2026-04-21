import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

class SentenceNormalizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        lemmatized_words = [self.lemmatize_word(word, tag) for word, tag in pos_tag(word_tokenize(self.remove_punctuation(sentence)))]
        return lemmatized_words

    def get_pos_tag(self, sentence):
        return [tag for word, tag in pos_tag(word_tokenize(self.remove_punctuation(sentence)))]
    
    def lemmatize_word(self, word, tag):
        pos_tag = tag[0].upper()
        return self.lemmatizer.lemmatize(word, pos=pos_tag) if pos_tag in ['V', 'J', 'R'] else self.lemmatizer.lemmatize(word)

    def remove_punctuation(self, sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
