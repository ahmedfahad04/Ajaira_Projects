note_map = {'o': 4, 'o|': 2, '.|': 1}
result = []
for note in music_string.split(' '):
    if note:
        result.append(note_map[note])
return result
