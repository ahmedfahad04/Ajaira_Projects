class AcousticGuitarSimulator:
        def __init__(self, text):
            self.text_input = text

        def analyze(self, output_flag=False):
            if not self.text_input.strip():
                return []
            else:
                notes_list = []
                segments = self.text_input.split(" ")
                for segment in segments:
                    non_alpha_pos = next((i for i, char in enumerate(segment) if not char.isalpha()), len(segment))
                    note_chord = segment[:non_alpha_pos]
                    note_tune = segment[non_alpha_pos:]
                    notes_list.append({'Chord': note_chord, 'Tune': note_tune})
                    if output_flag:
                        self.output_note(note_chord, note_tune)
                return notes_list

        def output_note(self, chord, tune):
            return f"Standard Acoustic Guitar Play -- Chord: {chord}, Tune: {tune}"
