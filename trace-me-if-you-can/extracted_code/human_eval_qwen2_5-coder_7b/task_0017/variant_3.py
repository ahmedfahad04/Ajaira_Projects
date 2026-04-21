note_translator = {'o': 4, 'o|': 2, '.|': 1}
def parse_notes(str):
    return [note_translator[note] for note in str.split(' ') if note]
