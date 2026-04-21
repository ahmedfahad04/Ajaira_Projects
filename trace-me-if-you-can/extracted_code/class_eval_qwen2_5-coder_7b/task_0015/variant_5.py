class BoyerMooreFinder:
    def __init__(self, document, keyword):
        self.document = document
        self.keyword = keyword
        self.doc_length = len(document)
        self.key_length = len(keyword)

    def char_matches_keyword(self, char):
        for index in range(self.key_length - 1, -1, -1):
            if char == self.keyword[index]:
                return index
        return -1

    def find_keyword_mismatch(self, pos):
        for index in range(self.key_length - 1, -1, -1):
            if self.keyword[index] != self.document[pos + index]:
                return pos + index
        return -1

    def apply_bad_char_heuristic(self):
        positions = []
        for pos in range(self.doc_length - self.key_length + 1):
            mismatch_pos = self.find_keyword_mismatch(pos)
            if mismatch_pos == -1:
                positions.append(pos)
            else:
                match_pos = self.char_matches_keyword(self.document[mismatch_pos])
                pos += (mismatch_pos - match_pos)
        return positions
