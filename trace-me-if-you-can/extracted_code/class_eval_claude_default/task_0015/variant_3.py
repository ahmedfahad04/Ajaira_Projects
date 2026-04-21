class BoyerMooreSearch:
    def __init__(self, text, pattern):
        self.text, self.pattern = text, pattern
        self.textLen, self.patLen = len(text), len(pattern)
        
    def find_rightmost_occurrence(self, char, pattern):
        return next((i for i in reversed(range(len(pattern))) if pattern[i] == char), -1)
    
    def find_first_mismatch(self, text_start):
        pattern_chars = self.pattern
        text_segment = self.text[text_start:text_start + self.patLen]
        
        mismatch_positions = [text_start + i for i, (p, t) in 
                            enumerate(zip(pattern_chars, text_segment)) if p != t]
        
        return mismatch_positions[0] if mismatch_positions else -1
    
    def bad_character_heuristic(self):
        found_positions = []
        current_position = 0
        
        while current_position <= self.textLen - self.patLen:
            first_mismatch_pos = self.find_first_mismatch(current_position)
            
            if first_mismatch_pos == -1:
                found_positions.append(current_position)
                current_position += 1
            else:
                mismatched_char = self.text[first_mismatch_pos]
                rightmost_pos = self.find_rightmost_occurrence(mismatched_char, self.pattern)
                current_position = max(current_position + 1, first_mismatch_pos - rightmost_pos)
                
        return found_positions
