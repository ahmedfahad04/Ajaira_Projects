def get_note_values(music_input):
    note_dict = {'o': 4, 'o|': 2, '.|': 1}
    return [note_dict[note] for note in music_input.split(' ') if note]
