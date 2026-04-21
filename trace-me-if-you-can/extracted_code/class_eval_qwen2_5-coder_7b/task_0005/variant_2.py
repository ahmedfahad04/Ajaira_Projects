class GuitarPlaySimulator:
        def __init__(self, text):
            self.input_text = text

        def execute(self, output_mode=False):
            if not self.input_text.strip():
                return []
            else:
                output_list = []
                elements = self.input_text.split(" ")
                for element in elements:
                    letter_index = next((i for i, char in enumerate(element) if not char.isalpha()), len(element))
                    note = element[:letter_index]
                    tune = element[letter_index:]
                    output_list.append({'Melody': note, 'Tune': tune})
                    if output_mode:
                        self.display_note(note, tune)
                return output_list

        def display_note(self, melody, tune):
            return f"Standard Guitar Performance -- Melody: {melody}, Tune: {tune}"
