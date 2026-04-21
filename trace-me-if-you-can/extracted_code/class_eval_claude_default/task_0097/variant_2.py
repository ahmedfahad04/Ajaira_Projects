class Words2Numbers:

    def __init__(self):
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build all lookup tables using a functional approach"""
        # Define base data
        base_numbers = [
            ("zero", 0), ("one", 1), ("two", 2), ("three", 3), ("four", 4),
            ("five", 5), ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9),
            ("ten", 10), ("eleven", 11), ("twelve", 12), ("thirteen", 13),
            ("fourteen", 14), ("fifteen", 15), ("sixteen", 16), ("seventeen", 17),
            ("eighteen", 18), ("nineteen", 19)
        ]
        
        tens_data = [("twenty", 20), ("thirty", 30), ("forty", 40), ("fifty", 50),
                    ("sixty", 60), ("seventy", 70), ("eighty", 80), ("ninety", 90)]
        
        scale_multipliers = [("hundred", 100), ("thousand", 1000), ("million", 1000000),
                           ("billion", 1000000000), ("trillion", 1000000000000)]
        
        # Build numwords dictionary
        self.numwords = {"and": (1, 0)}
        
        # Add units and base numbers
        for word, value in base_numbers:
            self.numwords[word] = (1, value)
        
        # Add tens
        for word, value in tens_data:
            self.numwords[word] = (1, value)
        
        # Add scales
        for word, multiplier in scale_multipliers:
            self.numwords[word] = (multiplier, 0)
        
        self.ordinal_words = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        self.ordinal_endings = [('ieth', 'y'), ('th', '')]

    def _process_word(self, word):
        """Process a single word and return its numeric interpretation"""
        if word in self.ordinal_words:
            return (1, self.ordinal_words[word]), True
        
        # Try ordinal endings
        processed_word = word
        for ending, replacement in self.ordinal_endings:
            if word.endswith(ending):
                processed_word = word[:-len(ending)] + replacement
                break
        
        if processed_word in self.numwords:
            return self.numwords[processed_word], True
        
        return None, False

    def text2int(self, textnum):
        tokens = textnum.replace('-', ' ').split()
        current = result = 0
        output_buffer = []
        processing_number = False
        
        for token in tokens:
            word_data, is_numeric = self._process_word(token)
            
            if is_numeric:
                scale, increment = word_data
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                processing_number = True
            else:
                if processing_number:
                    output_buffer.append(repr(result + current))
                output_buffer.append(token)
                result = current = 0
                processing_number = False
        
        if processing_number:
            output_buffer.append(repr(result + current))
        
        return ' '.join(output_buffer)

    def is_valid_input(self, textnum):
        tokens = textnum.replace('-', ' ').split()
        return all(self._process_word(token)[1] for token in tokens)
