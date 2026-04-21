import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string
from functools import partial

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')


class Lemmatization:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_remover = partial(str.translate, table=str.maketrans('', '', string.punctuation))

    def lemmatize_sentence(self, sentence):
        processed_sentence = self.punctuation_remover(sentence)
        tokens = word_tokenize(processed_sentence)
        pos_tagged = pos_tag(tokens)
        
        lemmatize_func = lambda word_tag: self._lemmatize_by_pos(*word_tag)
        return list(map(lemmatize_func, pos_tagged))

    def _lemmatize_by_pos(self, word, tag):
        pos_dict = {'V': 'v', 'J': 'a', 'R': 'r'}
        matched_pos = next((pos for prefix, pos in pos_dict.items() if tag.startswith(prefix)), None)
        
        if matched_pos:
            return self.lemmatizer.lemmatize(word, pos=matched_pos)
        return self.lemmatizer.lemmatize(word)

    def get_pos_tag(self, sentence):
        processed_sentence = self.punctuation_remover(sentence)
        tokens = word_tokenize(processed_sentence)
        pos_tagged = pos_tag(tokens)
        return [tag for word, tag in pos_tagged]

    def remove_punctuation(self, sentence):
        return self.punctuation_remover(sentence)
