class AutoGuitarSimulator:
        def __init__(self, text):
            self.sequence = text

        def process(self, display_output=False):
            if not self.sequence.strip():
                return []
            else:
                parsed_sequence = []
                segments = self.sequence.split(" ")
                for segment in segments:
                    note_index = next((i for i, char in enumerate(segment) if not char.isalpha()), len(segment))
                    note_name = segment[:note_index]
                    note_tune = segment[note_index:]
                    parsed_sequence.append({'Key': note_name, 'Tune': note_tune})
                    if display_output:
                        self.show_note(note_name, note_tune)
                return parsed_sequence

        def show_note(self, key, tune):
            return f"Standard Guitar Play -- Key: {key}, Tune: {tune}"
