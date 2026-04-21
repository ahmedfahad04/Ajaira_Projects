note_map = {'o': 4, 'o|': 2, '.|': 1}
notes = filter(bool, music_string.split(' '))
return list(map(note_map.__getitem__, notes))
