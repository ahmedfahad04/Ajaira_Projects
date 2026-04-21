note_map = {'o': 4, 'o|': 2, '.|': 1}
return list(filter(None, [note_map.get(x, 0) for x in music_string.split(' ')]))
