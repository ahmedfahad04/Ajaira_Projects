class BoyerMooreSearch:
    def __init__(self, text, pattern):
        self.text, self.pattern = text, pattern
        self.textLen, self.patLen = len(text), len(pattern)

    def match_in_pattern(self, char):
        try:
            return self.pattern.rindex(char)
        except ValueError:
            return -1

    def mismatch_in_text(self, currentPos):
        text_slice = self.text[currentPos:currentPos + self.patLen]
        for offset, (p_char, t_char) in enumerate(zip(self.pattern, text_slice)):
            if p_char != t_char:
                return currentPos + offset
        return -1

    def bad_character_heuristic(self):
        positions = []
        i = 0
        while i <= self.textLen - self.patLen:
            mismatch_index = self.mismatch_in_text(i)
            if mismatch_index == -1:
                positions.append(i)
                i += 1
            else:
                match_index = self.match_in_pattern(self.text[mismatch_index])
                shift = max(1, mismatch_index - match_index)
                i += shift
        return positions
