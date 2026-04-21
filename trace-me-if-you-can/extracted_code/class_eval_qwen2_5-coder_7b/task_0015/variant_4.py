class BoyerMooreSearchEngine:
    def __init__(self, data, query):
        self.data = data
        self.query = query
        self.data_length = len(data)
        self.query_length = len(query)

    def match_char_in_query(self, char):
        for index in range(self.query_length - 1, -1, -1):
            if char == self.query[index]:
                return index
        return -1

    def find_char_mismatch(self, pos):
        for index in range(self.query_length - 1, -1, -1):
            if self.query[index] != self.data[pos + index]:
                return pos + index
        return -1

    def execute_bad_char_heuristic(self):
        positions = []
        for pos in range(self.data_length - self.query_length + 1):
            mismatch_pos = self.find_char_mismatch(pos)
            if mismatch_pos == -1:
                positions.append(pos)
            else:
                match_pos = self.match_char_in_query(self.data[mismatch_pos])
                pos += (mismatch_pos - match_pos)
        return positions
