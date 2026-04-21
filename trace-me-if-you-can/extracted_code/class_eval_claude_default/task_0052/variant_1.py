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
        self.pos_mapping = {
            'V': 'v',
            'J': 'a', 
            'R': 'r'
        }

    def lemmatize_sentence(self, sentence):
        clean_sentence = self.remove_punctuation(sentence)
        tagged_words = pos_tag(word_tokenize(clean_sentence))
        
        return [self._get_lemmatized_word(word, tag) for word, tag in tagged_words]

    def _get_lemmatized_word(self, word, tag):
        pos_key = next((key for key in self.pos_mapping if tag.startswith(key)), None)
        if pos_key:
            return self.lemmatizer.lemmatize(word, pos=self.pos_mapping[pos_key])
        return self.lemmatizer.lemmatize(word)

    def get_pos_tag(self, sentence):
        clean_sentence = self.remove_punctuation(sentence)
        tagged_words = pos_tag(word_tokenize(clean_sentence))
        return [tag for _, tag in tagged_words]

    def remove_punctuation(self, sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
