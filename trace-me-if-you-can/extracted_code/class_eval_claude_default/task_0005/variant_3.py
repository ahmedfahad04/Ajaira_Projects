from itertools import takewhile

class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        self.play_text = text

    def interpret(self, display=False):
        stripped_text = self.play_text.strip()
        if not stripped_text:
            return []
        
        def process_segment(segment):
            chord_chars = list(takewhile(str.isalpha, segment))
            chord = ''.join(chord_chars)
            tune = segment[len(chord_chars):]
            
            play_entry = {'Chord': chord, 'Tune': tune}
            if display:
                self.display(chord, tune)
            return play_entry
        
        return list(map(process_segment, stripped_text.split(" ")))

    def display(self, key, value):
        return "Normal Guitar Playing -- Chord: %s, Play Tune: %s" % (key, value)
