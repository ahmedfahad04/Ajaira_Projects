def convert_notes(music_str):
    note_map = {'o': 4, 'o|': 2, '.|': 1}
    return [note_map[note] for note in music_str.split(' ') if note]
