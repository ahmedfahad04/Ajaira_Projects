class BMHSearch:
    def __init__(self, text, pattern):
        self.text = text
        self.pattern = pattern
        self.text_length = len(text)
        self.pattern_length = len(pattern)

    def character_matches_pattern(self, character):
        for index in range(self.pattern_length - 1, -1, -1):
            if character == self.pattern[index]:
                return index
        return -1

    def find_mismatch(self, current_position):
        for index in range(self.pattern_length - 1, -1, -1):
            if self.pattern[index] != self.text[current_position + index]:
                return current_position + index
        return -1

    def execute_bad_char_heuristic(self):
        positions = []
        for index in range(self.text_length - self.pattern_length + 1):
            mismatch_index = self.find_mismatch(index)
            if mismatch_index == -1:
                positions.append(index)
            else:
                match_index = self.character_matches_pattern(self.text[mismatch_index])
                index += (mismatch_index - match_index)
        return positions
