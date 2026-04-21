class BoyerMooreSearch:
    def __init__(self, text, pattern):
        self.text, self.pattern = text, pattern
        self.textLen, self.patLen = len(text), len(pattern)
        self._pattern_char_positions = self._compute_char_positions()

    def _compute_char_positions(self):
        positions = {}
        for idx, char in enumerate(self.pattern):
            if char not in positions:
                positions[char] = []
            positions[char].append(idx)
        return positions

    def match_in_pattern(self, char):
        if char in self._pattern_char_positions:
            return self._pattern_char_positions[char][-1]
        return -1

    def mismatch_in_text(self, currentPos):
        if currentPos + self.patLen > self.textLen:
            return currentPos
            
        comparison_pairs = list(zip(
            range(self.patLen), 
            self.pattern, 
            self.text[currentPos:currentPos + self.patLen]
        ))
        
        for offset, p_char, t_char in comparison_pairs:
            if p_char != t_char:
                return currentPos + offset
        return -1

    def bad_character_heuristic(self):
        positions = []
        pos = 0
        
        while pos <= self.textLen - self.patLen:
            mismatch_index = self.mismatch_in_text(pos)
            if mismatch_index == -1:
                positions.append(pos)
                pos += 1
            else:
                match_index = self.match_in_pattern(self.text[mismatch_index])
                jump_distance = mismatch_index - match_index
                pos = jump_distance if jump_distance > 0 else pos + 1
        return positions
