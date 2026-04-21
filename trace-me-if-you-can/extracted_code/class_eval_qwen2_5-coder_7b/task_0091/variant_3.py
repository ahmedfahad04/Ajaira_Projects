import urllib.parse


    class URLSegmentCollector:
        def __init__(self):
            self.segs = []
            self.ends_with_slash = False

        def append_segment(self, segment):
            self.segs.append(self._sanitize_segment(segment))

        def dissect_path(self, path, charset):
            if path:
                if path.endswith('/'):
                    self.ends_with_slash = True

                sanitized_path = self._sanitize_segment(path)
                if sanitized_path:
                    segments = sanitized_path.split('/')
                    for seg in segments:
                        decoded_segment = urllib.parse.unquote(seg, encoding=charset)
                        self.segs.append(decoded_segment)

        @staticmethod
        def _sanitize_segment(segment):
            if not segment:
                return ''

            sanitized_segment = segment.strip('/')
            return sanitized_segment
