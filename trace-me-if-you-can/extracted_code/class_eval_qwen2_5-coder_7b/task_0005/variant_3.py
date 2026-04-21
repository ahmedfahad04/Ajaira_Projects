class ElectricGuitarSimulator:
        def __init__(self, text):
            self.input_sequence = text

        def run(self, should_display=False):
            if not self.input_sequence.strip():
                return []
            else:
                result = []
                elements = self.input_sequence.split(" ")
                for element in elements:
                    non_alpha_index = next((i for i, char in enumerate(element) if not char.isalpha()), len(element))
                    note_name = element[:non_alpha_index]
                    tune = element[non_alpha_index:]
                    result.append({'Note': note_name, 'Tune': tune})
                    if should_display:
                        self.show_note(note_name, tune)
                return result

        def show_note(self, note, tune):
            return f"Standard Electric Guitar Play -- Note: {note}, Tune: {tune}"
