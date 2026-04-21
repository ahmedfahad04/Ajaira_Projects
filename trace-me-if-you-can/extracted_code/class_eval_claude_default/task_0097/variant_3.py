from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

class WordType(Enum):
    UNIT = "unit"
    SCALE = "scale"
    ORDINAL = "ordinal"
    INVALID = "invalid"

@dataclass
class NumberWord:
    word_type: WordType
    scale: int
    increment: int

class Words2Numbers:

    def __init__(self):
        self.word_registry = self._create_word_registry()
        self.ordinal_transformations = [('ieth', 'y'), ('th', '')]

    def _create_word_registry(self):
        """Create a registry of all number words with their properties"""
        registry = {}
        
        # Units (0-19)
        unit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                     "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                     "sixteen", "seventeen", "eighteen", "nineteen"]
        
        for i, name in enumerate(unit_names):
            registry[name] = NumberWord(WordType.UNIT, 1, i)
        
        # Tens (20, 30, 40, ...)
        tens_names = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        for i, name in enumerate(tens_names):
            registry[name] = NumberWord(WordType.UNIT, 1, (i + 2) * 10)
        
        # Scales
        scale_data = [("hundred", 100), ("thousand", 1000), ("million", 1000000), 
                     ("billion", 1000000000), ("trillion", 1000000000000)]
        for name, multiplier in scale_data:
            registry[name] = NumberWord(WordType.SCALE, multiplier, 0)
        
        # Special cases
        registry["and"] = NumberWord(WordType.UNIT, 1, 0)
        
        # Ordinals
        ordinal_map = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        for word, value in ordinal_map.items():
            registry[word] = NumberWord(WordType.ORDINAL, 1, value)
        
        return registry

    def _resolve_word(self, word: str) -> Optional[NumberWord]:
        """Resolve a word to its NumberWord representation"""
        if word in self.word_registry:
            return self.word_registry[word]
        
        # Try ordinal transformations
        for suffix, replacement in self.ordinal_transformations:
            if word.endswith(suffix):
                transformed = word[:-len(suffix)] + replacement
                if transformed in self.word_registry:
                    return self.word_registry[transformed]
        
        return None

    def text2int(self, textnum):
        word_sequence = textnum.replace('-', ' ').split()
        accumulator = total = 0
        result_fragments = []
        in_number_context = False
        
        for word in word_sequence:
            number_word = self._resolve_word(word)
            
            if number_word:
                accumulator = accumulator * number_word.scale + number_word.increment
                if number_word.scale > 100:
                    total += accumulator
                    accumulator = 0
                in_number_context = True
            else:
                if in_number_context:
                    result_fragments.append(repr(total + accumulator))
                result_fragments.append(word)
                total = accumulator = 0
                in_number_context = False
        
        if in_number_context:
            result_fragments.append(repr(total + accumulator))
        
        return ' '.join(result_fragments)

    def is_valid_input(self, textnum):
        word_sequence = textnum.replace('-', ' ').split()
        return all(self._resolve_word(word) is not None for word in word_sequence)
