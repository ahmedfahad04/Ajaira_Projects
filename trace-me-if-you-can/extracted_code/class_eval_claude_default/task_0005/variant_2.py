import re

class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        self.play_text = text

    def interpret(self, display=False):
        text = self.play_text.strip()
        if not text:
            return []
        
        segments = text.split(" ")
        return [self._parse_segment(seg, display) for seg in segments]
    
    def _parse_segment(self, segment, should_display):
        match = re.match(r'^([a-zA-Z]*)(.*)$', segment)
        chord, tune = match.groups() if match else ('', segment)
        
        result = {'Chord': chord, 'Tune': tune}
        if should_display:
            self.display(chord, tune)
        
        return result

    def display(self, key, value):
        return "Normal Guitar Playing -- Chord: %s, Play Tune: %s" % (key, value)
