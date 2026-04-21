class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        self.play_text = text

    def interpret(self, display=False):
        segments = self.play_text.strip().split(" ")
        
        if not segments or segments == ['']:
            return []
        
        result = []
        for segment in segments:
            chord_tune_pair = self._extract_chord_and_tune(segment)
            result.append(chord_tune_pair)
            
            if display:
                self.display(chord_tune_pair['Chord'], chord_tune_pair['Tune'])
        
        return result
    
    def _extract_chord_and_tune(self, segment):
        boundary = next((i for i, c in enumerate(segment) if not c.isalpha()), len(segment))
        return {
            'Chord': segment[:boundary],
            'Tune': segment[boundary:]
        }

    def display(self, key, value):
        return "Normal Guitar Playing -- Chord: %s, Play Tune: %s" % (key, value)
