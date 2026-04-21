def extract_note_values(music_data):
    note_codes = {'o': 4, 'o|': 2, '.|': 1}
    return [note_codes[note] for note in music_data.split(' ') if note]
