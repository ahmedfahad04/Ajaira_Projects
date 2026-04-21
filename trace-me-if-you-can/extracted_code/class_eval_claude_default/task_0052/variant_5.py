import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string
from itertools import starmap

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')


class Lemmatization:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        tagged_tokens = self._tokenize_and_tag(sentence)
        lemmatize_token = self._create_lemmatizer()
        return list(starmap(lemmatize_token, tagged_tokens))

    def _tokenize_and_tag(self, sentence):
        cleaned = self.remove_punctuation(sentence)
        tokens = word_tokenize(cleaned)
        return pos_tag(tokens)

    def _create_lemmatizer(self):
        def lemmatize_with_pos(word, tag):
            tag_to_pos = {
                ('V',): 'v',
                ('J',): 'a', 
                ('R',): 'r'
            }
            
            wordnet_pos = next(
                (pos for prefixes, pos in tag_to_pos.items() if any(tag.startswith(p) for p in prefixes)),
                None
            )
            
            return (self.lemmatizer.lemmatize(word, pos=wordnet_pos) 
                   if wordnet_pos 
                   else self.lemmatizer.lemmatize(word))
        
        return lemmatize_with_pos

    def get_pos_tag(self, sentence):
        tagged_tokens = self._tokenize_and_tag(sentence)
        return [tag for word, tag in tagged_tokens]

    def remove_punctuation(self, sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
