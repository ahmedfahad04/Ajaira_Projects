from functools import lru_cache

class BoyerMooreSearch:
    def __init__(self, text, pattern):
        self.text, self.pattern = text, pattern
        self.textLen, self.patLen = len(text), len(pattern)

    @lru_cache(maxsize=256)
    def match_in_pattern(self, char):
        indices = [i for i, c in enumerate(self.pattern) if c == char]
        return max(indices) if indices else -1

    def mismatch_in_text(self, currentPos):
        end_pos = min(currentPos + self.patLen, self.textLen)
        text_window = self.text[currentPos:end_pos]
        
        for i, (pattern_char, text_char) in enumerate(zip(self.pattern, text_window)):
            if pattern_char != text_char:
                return currentPos + i
        return -1

    def bad_character_heuristic(self):
        positions = []
        search_index = 0
        
        while search_index <= self.textLen - self.patLen:
            mismatch_location = self.mismatch_in_text(search_index)
            
            if mismatch_location == -1:
                positions.append(search_index)
                search_index += 1
            else:
                bad_char = self.text[mismatch_location]
                pattern_match_pos = self.match_in_pattern(bad_char)
                search_index = mismatch_location - pattern_match_pos if pattern_match_pos != -1 else mismatch_location + 1
                
        return positions
