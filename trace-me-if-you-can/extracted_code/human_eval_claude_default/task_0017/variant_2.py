note_map = {'o': 4, 'o|': 2, '.|': 1}
return list(note_map[note] for note in music_string.split(' ') if note in note_map)
