class BoyerMoore:
    def __init__(self, source, search_term):
        self.source = source
        self.search_term = search_term
        self.source_length = len(source)
        self.search_term_length = len(search_term)

    def does_char_match(self, char):
        for i in range(self.search_term_length - 1, -1, -1):
            if char == self.search_term[i]:
                return i
        return -1

    def locate_mismatch(self, pos):
        for i in range(self.search_term_length - 1, -1, -1):
            if self.search_term[i] != self.source[pos + i]:
                return pos + i
        return -1

    def execute_bad_char_rule(self):
        positions = []
        for pos in range(self.source_length - self.search_term_length + 1):
            mismatch_pos = self.locate_mismatch(pos)
            if mismatch_pos == -1:
                positions.append(pos)
            else:
                match_pos = self.does_char_match(self.source[mismatch_pos])
                pos += (mismatch_pos - match_pos)
        return positions
