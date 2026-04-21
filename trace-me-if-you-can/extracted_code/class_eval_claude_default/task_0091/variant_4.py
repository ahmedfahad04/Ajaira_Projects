import urllib.parse


class UrlPath:
    def __init__(self):
        self.segments = []
        self.with_end_tag = False

    def add(self, segment):
        fixed_segment = segment.strip('/') if segment else ''
        self.segments.append(fixed_segment)

    def parse(self, path, charset):
        if path:
            self.with_end_tag = path[-1] == '/'
            
            trimmed_path = path.strip('/')
            if trimmed_path:
                segment_list = trimmed_path.split('/')
                for segment in segment_list:
                    decoded = urllib.parse.unquote(segment, encoding=charset)
                    self.segments.append(decoded)

    @staticmethod
    def fix_path(path):
        return path.strip('/') if path else ''
