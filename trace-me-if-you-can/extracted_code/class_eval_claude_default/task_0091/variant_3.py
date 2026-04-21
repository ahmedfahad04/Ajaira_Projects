import urllib.parse


class UrlPath:
    def __init__(self):
        self.segments = []
        self.with_end_tag = False

    def add(self, segment):
        self.segments.append(self.fix_path(segment))

    def parse(self, path, charset):
        self.with_end_tag = bool(path and path.endswith('/'))
        
        segments_to_add = self._get_decoded_segments(path, charset)
        self.segments.extend(segments_to_add)

    def _get_decoded_segments(self, path, charset):
        if not path:
            return []
            
        cleaned_path = self.fix_path(path)
        if not cleaned_path:
            return []
            
        raw_segments = cleaned_path.split('/')
        return [urllib.parse.unquote(seg, encoding=charset) for seg in raw_segments]

    @staticmethod
    def fix_path(path):
        if not path:
            return ''
        return path.strip('/')
