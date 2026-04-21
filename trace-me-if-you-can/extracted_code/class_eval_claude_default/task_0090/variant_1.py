import re

class URLHandler:
    def __init__(self, url):
        self.url = url
        self._parsed = self._parse_url()
    
    def _parse_url(self):
        pattern = r'^(?:([^:/?#]+)://)?([^/?#]*)([^?#]*)(?:\?([^#]*))?(?:#(.*))?$'
        match = re.match(pattern, self.url)
        if match:
            return {
                'scheme': match.group(1),
                'host': match.group(2),
                'path': match.group(3),
                'query': match.group(4),
                'fragment': match.group(5)
            }
        return {}

    def get_scheme(self):
        return self._parsed.get('scheme')

    def get_host(self):
        return self._parsed.get('host') or None

    def get_path(self):
        path = self._parsed.get('path')
        return path if path else None

    def get_query_params(self):
        query = self._parsed.get('query')
        if query:
            params = {}
            for pair in query.split("&"):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    params[key] = value
            return params
        return None

    def get_fragment(self):
        return self._parsed.get('fragment')
