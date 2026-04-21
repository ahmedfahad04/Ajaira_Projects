class BoyerMooreSearch:
    def __init__(self, text, pattern):
        self.text = text
        self.pattern = pattern
        self.textLen = len(text)
        self.patLen = len(pattern)
        # Pre-compute bad character table for efficiency
        self.bad_char_table = self._build_bad_char_table()
    
    def _build_bad_char_table(self):
        table = {}
        for i, char in enumerate(self.pattern):
            table[char] = i
        return table
    
    def _get_last_occurrence(self, char):
        return self.bad_char_table.get(char, -1)
    
    def _check_pattern_match(self, start_pos):
        for i in range(self.patLen):
            if self.pattern[i] != self.text[start_pos + i]:
                return start_pos + i
        return -1
    
    def bad_character_heuristic(self):
        matches = []
        i = 0
        while i <= self.textLen - self.patLen:
            mismatch_pos = self._check_pattern_match(i)
            if mismatch_pos == -1:
                matches.append(i)
                i += 1
            else:
                mismatch_char = self.text[mismatch_pos]
                last_occurrence = self._get_last_occurrence(mismatch_char)
                i = max(1, mismatch_pos - last_occurrence)
        return matches
