import urllib.parse


    class URLPath:
        def __init__(self):
            self.path_segments = []
            self.has_trailing_slash = False

        def add_segment(self, segment):
            self.path_segments.append(self._normalize_segment(segment))

        def parse_path(self, path, charset):
            if path:
                if path.endswith('/'):
                    self.has_trailing_slash = True

                normalized_path = self._normalize_segment(path)
                if normalized_path:
                    segments = normalized_path.split('/')
                    for seg in segments:
                        decoded_segment = urllib.parse.unquote(seg, encoding=charset)
                        self.path_segments.append(decoded_segment)

        @staticmethod
        def _normalize_segment(segment):
            if not segment:
                return ''

            trimmed_segment = segment.strip('/')
            return trimmed_segment
