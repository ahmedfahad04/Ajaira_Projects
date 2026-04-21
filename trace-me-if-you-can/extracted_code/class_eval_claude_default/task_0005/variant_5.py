class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        self.play_text = text

    def interpret(self, display=False):
        normalized_text = self.play_text.strip()
        
        # Early return for empty input
        play_list = [] if not normalized_text else self._build_play_list(normalized_text, display)
        return play_list
    
    def _build_play_list(self, text, show_display):
        play_segments = text.split(" ")
        play_list = []
        
        for segment in play_segments:
            chord, tune = self._split_chord_tune(segment)
            play_dict = self._create_play_entry(chord, tune)
            play_list.append(play_dict)
            
            show_display and self.display(chord, tune)
        
        return play_list
    
    def _split_chord_tune(self, segment):
        alpha_count = sum(1 for char in segment if char.isalpha() and segment.index(char) == segment[:segment.index(char)+1].count(char) - 1 + segment[:segment.index(char)].count(char))
        # Simplified: count consecutive alphabetic characters from start
        alpha_end = 0
        for char in segment:
            if char.isalpha():
                alpha_end += 1
            else:
                break
        
        return segment[:alpha_end], segment[alpha_end:]
    
    def _create_play_entry(self, chord, tune):
        return {'Chord': chord, 'Tune': tune}

    def display(self, key, value):
        return "Normal Guitar Playing -- Chord: %s, Play Tune: %s" % (key, value)
