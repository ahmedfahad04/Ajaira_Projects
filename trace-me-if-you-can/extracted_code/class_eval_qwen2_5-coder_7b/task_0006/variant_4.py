class Segmenter:
    def __init__(self, series, segment_size):
        self.series = series
        self.segment_size = segment_size

    def calculate_segment_size(self):
        basic_size = len(self.series) // self.segment_size
        extra_segments = len(self.series) % self.segment_size
        return basic_size, extra_segments

    def get_segment(self, segment_number):
        basic_size, extra_segments = self.calculate_segment_size()
        start = segment_number * basic_size + min(segment_number, extra_segments)
        end = start + basic_size
        if segment_number + 1 <= extra_segments:
            end += 1
        return self.series[start:end]
