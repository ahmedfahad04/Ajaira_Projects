import urllib.parse


    class PathAggregator:
        def __init__(self):
            self.path_parts = []
            self.has_trailing_slash = False

        def add_path_part(self, segment):
            self.path_parts.append(self._clean_segment(segment))

        def extract_path(self, path, charset):
            if path:
                if path.endswith('/'):
                    self.has_trailing_slash = True

                cleaned_path = self._clean_segment(path)
                if cleaned_path:
                    parts = cleaned_path.split('/')
                    for part in parts:
                        decoded_part = urllib.parse.unquote(part, encoding=charset)
                        self.path_parts.append(decoded_part)

        @staticmethod
        def _clean_segment(segment):
            if not segment:
                return ''

            cleaned_segment = segment.strip('/')
            return cleaned_segment
