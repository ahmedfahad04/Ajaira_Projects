import urllib.parse
import functools


class UrlPath:
    def __init__(self):
        self.segments = []
        self.with_end_tag = False

    def add(self, segment):
        self.segments.append(self.fix_path(segment))

    def parse(self, path, charset):
        path_processor = functools.partial(self._process_path, charset=charset)
        path_processor(path)

    def _process_path(self, path, charset):
        if not path:
            return
            
        self.with_end_tag = path.endswith('/')
        normalized = self.fix_path(path)
        
        if normalized:
            segments = map(lambda s: urllib.parse.unquote(s, encoding=charset), 
                          normalized.split('/'))
            self.segments.extend(segments)

    @staticmethod
    def fix_path(path):
        if not path:
            return ''
        segment_str = path.strip('/')
        return segment_str
