class Words2Numbers:
    
    def __init__(self):
        # Use dictionary comprehensions and zip for more concise initialization
        self.units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                     "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                     "sixteen", "seventeen", "eighteen", "nineteen"]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.scales = ["hundred", "thousand", "million", "billion", "trillion"]
        
        # Build numwords using dictionary comprehension and merging
        self.numwords = {"and": (1, 0)}
        self.numwords.update({word: (1, idx) for idx, word in enumerate(self.units)})
        self.numwords.update({word: (1, idx * 10) for idx, word in enumerate(self.tens) if word})
        self.numwords.update({word: (10 ** (idx * 3 or 2), 0) for idx, word in enumerate(self.scales)})
        
        self.ordinal_words = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        self.ordinal_endings = [('ieth', 'y'), ('th', '')]

    def _normalize_ordinal(self, word):
        """Extract ordinal normalization logic into separate method"""
        for ending, replacement in self.ordinal_endings:
            if word.endswith(ending):
                return word[:-len(ending)] + replacement
        return word

    def text2int(self, textnum):
        words = textnum.replace('-', ' ').split()
        current = result = 0
        curstring = ""
        onnumber = False
        
        for word in words:
            if word in self.ordinal_words:
                current = current * 1 + self.ordinal_words[word]
                onnumber = True
            else:
                normalized_word = self._normalize_ordinal(word)
                
                if normalized_word not in self.numwords:
                    if onnumber:
                        curstring += repr(result + current) + " "
                    curstring += word + " "
                    result = current = 0
                    onnumber = False
                else:
                    scale, increment = self.numwords[normalized_word]
                    current = current * scale + increment
                    if scale > 100:
                        result += current
                        current = 0
                    onnumber = True

        return curstring + repr(result + current) if onnumber else curstring

    def is_valid_input(self, textnum):
        words = textnum.replace('-', ' ').split()
        return all(word in self.ordinal_words or self._normalize_ordinal(word) in self.numwords 
                  for word in words)
