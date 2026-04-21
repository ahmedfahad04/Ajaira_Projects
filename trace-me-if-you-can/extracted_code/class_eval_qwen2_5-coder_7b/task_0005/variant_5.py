class DigitalGuitarSimulator:
        def __init__(self, text):
            self.input_string = text

        def play(self, display_enabled=False):
            if not self.input_string.strip():
                return []
            else:
                note_data = []
                segments = self.input_string.split(" ")
                for segment in segments:
                    non_alpha_pos = next((i for i, char in enumerate(segment) if not char.isalpha()), len(segment))
                    note = segment[:non_alpha_pos]
                    tune = segment[non_alpha_pos:]
                    note_data.append({'Note': note, 'Tune': tune})
                    if display_enabled:
                        self.show_note(note, tune)
                return note_data

        def show_note(self, note, tune):
            return f"Standard Digital Guitar Performance -- Note: {note}, Tune: {tune}"
