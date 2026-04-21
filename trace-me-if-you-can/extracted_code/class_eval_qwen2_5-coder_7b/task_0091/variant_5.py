import urllib.parse


    class URLPathHandler:
        def __init__(self):
            self.path_components = []
            self.has_trailing_slash = False

        def insert_segment(self, segment):
            self.path_components.append(self._trim_segment(segment))

        def parse_url(self, path, charset):
            if path:
                if path.endswith('/'):
                    self.has_trailing_slash = True

                trimmed_path = self._trim_segment(path)
                if trimmed_path:
                    segments = trimmed_path.split('/')
                    for seg in segments:
                        decoded_segment = urllib.parse.unquote(seg, encoding=charset)
                        self.path_components.append(decoded_segment)

        @staticmethod
        def _trim_segment(segment):
            if not segment:
                return ''

            trimmed_segment = segment.strip('/')
            return trimmed_segment
