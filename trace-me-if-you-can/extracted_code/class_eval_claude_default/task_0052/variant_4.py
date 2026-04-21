import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')


class Lemmatization:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        lemmatized_words = []
        
        for word, tag in self._get_tagged_words(sentence):
            if tag[0] in 'VJR':
                pos_mapping = {'V': 'v', 'J': 'a', 'R': 'r'}
                lemmatized_word = self.lemmatizer.lemmatize(word, pos=pos_mapping[tag[0]])
            else:
                lemmatized_word = self.lemmatizer.lemmatize(word)
            lemmatized_words.append(lemmatized_word)
            
        return lemmatized_words

    def _get_tagged_words(self, sentence):
        clean_sentence = self.remove_punctuation(sentence)
        words = word_tokenize(clean_sentence)
        return pos_tag(words)

    def get_pos_tag(self, sentence):
        tagged_words = self._get_tagged_words(sentence)
        pos_tags = []
        
        for _, tag in tagged_words:
            pos_tags.append(tag)
            
        return pos_tags

    def remove_punctuation(self, sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
