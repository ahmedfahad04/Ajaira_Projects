class BoyerMooreAlgorithm:
    def __init__(self, main_string, substring):
        self.main_string = main_string
        self.substring = substring
        self.main_length = len(main_string)
        self.substring_length = len(substring)

    def check_character_match(self, character):
        for index in range(self.substring_length - 1, -1, -1):
            if character == self.substring[index]:
                return index
        return -1

    def identify_mismatch(self, position):
        for index in range(self.substring_length - 1, -1, -1):
            if self.substring[index] != self.main_string[position + index]:
                return position + index
        return -1

    def run_bad_char_heuristic(self):
        positions = []
        for position in range(self.main_length - self.substring_length + 1):
            mismatch_position = self.identify_mismatch(position)
            if mismatch_position == -1:
                positions.append(position)
            else:
                match_position = self.check_character_match(self.main_string[mismatch_position])
                position += (mismatch_position - match_position)
        return positions
