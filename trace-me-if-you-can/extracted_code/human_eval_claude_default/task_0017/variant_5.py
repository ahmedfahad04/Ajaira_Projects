note_map = {'o': 4, 'o|': 2, '.|': 1}
non_empty_notes = [x for x in music_string.split(' ') if x]
return [note_map[note] for note in non_empty_notes]
