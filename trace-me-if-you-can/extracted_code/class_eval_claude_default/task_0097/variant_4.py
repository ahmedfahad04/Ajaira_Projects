class Words2Numbers:

    def __init__(self):
        self.lexicon = self._build_lexicon()

    def _build_lexicon(self):
        """Build the complete word-to-number lexicon using generators and iterators"""
        lexicon = {}
        
        # Generator for units
        def unit_generator():
            names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                    "sixteen", "seventeen", "eighteen", "nineteen"]
            for idx, name in enumerate(names):
                yield name, (1, idx)
        
        # Generator for tens
        def tens_generator():
            names = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            for idx, name in enumerate(names):
                yield name, (1, (idx + 2) * 10)
        
        # Generator for scales
        def scale_generator():
            scales = [("hundred", 2), ("thousand", 3), ("million", 6), ("billion", 9), ("trillion", 12)]
            for name, power in scales:
                yield name, (10 ** power, 0)
        
        # Build lexicon from generators
        lexicon.update(dict(unit_generator()))
        lexicon.update(dict(tens_generator()))
        lexicon.update(dict(scale_generator()))
        lexicon["and"] = (1, 0)
        
        return lexicon

    def _get_ordinal_mappings(self):
        """Return ordinal word mappings"""
        return {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}

    def _transform_ordinal_suffix(self, word):
        """Transform ordinal suffixes to cardinal form"""
        transformations = [('ieth', 'y'), ('th', '')]
        for old_suffix, new_suffix in transformations:
            if word.endswith(old_suffix):
                return word[:-len(old_suffix)] + new_suffix
        return word

    def _parse_single_word(self, word):
        """Parse a single word and return scale, increment, and validity"""
        ordinal_map = self._get_ordinal_mappings()
        
        if word in ordinal_map:
            return 1, ordinal_map[word], True
        
        transformed_word = self._transform_ordinal_suffix(word)
        if transformed_word in self.lexicon:
            return self.lexicon[transformed_word][0], self.lexicon[transformed_word][1], True
        
        return 0, 0, False

    def text2int(self, textnum):
        tokens = list(filter(None, textnum.replace('-', ' ').split()))
        
        running_total = partial_sum = 0
        output_segments = []
        number_in_progress = False
        
        for token in tokens:
            scale_factor, increment_value, is_valid = self._parse_single_word(token)
            
            if is_valid:
                partial_sum = partial_sum * scale_factor + increment_value
                if scale_factor > 100:
                    running_total += partial_sum
                    partial_sum = 0
                number_in_progress = True
            else:
                if number_in_progress:
                    output_segments.append(repr(running_total + partial_sum))
                output_segments.append(token)
                running_total = partial_sum = 0
                number_in_progress = False
        
        if number_in_progress:
            output_segments.append(repr(running_total + partial_sum))
        
        return ' '.join(output_segments)

    def is_valid_input(self, textnum):
        tokens = textnum.replace('-', ' ').split()
        validity_checks = [self._parse_single_word(token)[2] for token in tokens]
        return all(validity_checks)
