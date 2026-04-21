class Words2Numbers:
    
    def __init__(self):
        # Use dictionary comprehensions and zip for more concise initialization
        units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = ["hundred", "thousand", "million", "billion", "trillion"]
        
        self.numwords = {"and": (1, 0)}
        self.numwords.update({word: (1, idx) for idx, word in enumerate(units)})
        self.numwords.update({word: (1, idx * 10) for idx, word in enumerate(tens) if word})
        self.numwords.update({word: (10 ** (idx * 3 or 2), 0) for idx, word in enumerate(scales)})
        
        self.ordinal_words = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        self.ordinal_endings = [('ieth', 'y'), ('th', '')]

    def _normalize_ordinal(self, word):
        """Convert ordinal word to cardinal form"""
        if word in self.ordinal_words:
            return word, self.ordinal_words[word]
        
        for ending, replacement in self.ordinal_endings:
            if word.endswith(ending):
                normalized = word[:-len(ending)] + replacement
                return normalized, None
        
        return word, None

    def text2int(self, textnum):
        words = textnum.replace('-', ' ').split()
        current = result = 0
        output_parts = []
        onnumber = False
        
        for word in words:
            normalized_word, ordinal_value = self._normalize_ordinal(word)
            
            if ordinal_value is not None:
                current = current * 1 + ordinal_value
                onnumber = True
            elif normalized_word in self.numwords:
                scale, increment = self.numwords[normalized_word]
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True
            else:
                if onnumber:
                    output_parts.append(repr(result + current))
                output_parts.append(word)
                result = current = 0
                onnumber = False
        
        if onnumber:
            output_parts.append(repr(result + current))
        
        return ' '.join(output_parts)

    def is_valid_input(self, textnum):
        words = textnum.replace('-', ' ').split()
        return all(self._normalize_ordinal(word)[0] in self.numwords or word in self.ordinal_words 
                  for word in words)
