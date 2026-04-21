class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        self.play_text = text

    def interpret(self, display=False):
        if not self.play_text.strip():
            return []
        
        play_list = []
        for play_seg in self.play_text.split(" "):
            chord_end_idx = self._find_chord_end(play_seg)
            chord = play_seg[:chord_end_idx]
            tune = play_seg[chord_end_idx:]
            
            entry = {'Chord': chord, 'Tune': tune}
            play_list.append(entry)
            
            if display:
                self.display(chord, tune)
        
        return play_list
    
    def _find_chord_end(self, segment):
        for i, char in enumerate(segment):
            if not char.isalpha():
                return i
        return len(segment)

    def display(self, key, value):
        return "Normal Guitar Playing -- Chord: %s, Play Tune: %s" % (key, value)
