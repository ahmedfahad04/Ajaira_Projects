music_notes = {'o': 4, 'o|': 2, '.|': 1}
def translate_music(str):
    return [music_notes[note] for note in str.split(' ') if note]
