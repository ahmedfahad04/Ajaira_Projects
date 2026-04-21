import urllib.parse


    class PathManager:
        def __init__(self):
            self.sections = []
            self.terminated_with_slash = False

        def incorporate_segment(self, segment):
            self.sections.append(self._sanitize_segment(segment))

        def analyze_path(self, path, charset):
            if path:
                if path.endswith('/'):
                    self.terminated_with_slash = True

                sanitized_path = self._sanitize_segment(path)
                if sanitized_path:
                    segments = sanitized_path.split('/')
                    for seg in segments:
                        decoded_segment = urllib.parse.unquote(seg, encoding=charset)
                        self.sections.append(decoded_segment)

        @staticmethod
        def _sanitize_segment(segment):
            if not segment:
                return ''

            sanitized_segment = segment.strip('/')
            return sanitized_segment
