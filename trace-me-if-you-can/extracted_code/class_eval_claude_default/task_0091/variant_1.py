import urllib.parse


class UrlPath:
    def __init__(self):
        self.segments = []
        self.with_end_tag = False

    def add(self, segment):
        self.segments.append(self.fix_path(segment))

    def parse(self, path, charset):
        if not path:
            return
            
        self.with_end_tag = path.endswith('/')
        cleaned_path = self.fix_path(path)
        
        if cleaned_path:
            self.segments.extend(
                urllib.parse.unquote(seg, encoding=charset) 
                for seg in cleaned_path.split('/')
            )

    @staticmethod
    def fix_path(path):
        return path.strip('/') if path else ''
