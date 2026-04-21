class URLHandler:
    def __init__(self, url):
        self.url = url

    def _find_boundaries(self):
        return {
            'scheme_sep': self.url.find('://'),
            'path_start': self.url.find('/', self.url.find('://') + 3) if '://' in self.url else -1,
            'query_start': self.url.find('?'),
            'fragment_start': self.url.find('#')
        }

    def get_scheme(self):
        boundaries = self._find_boundaries()
        scheme_sep = boundaries['scheme_sep']
        return self.url[:scheme_sep] if scheme_sep > 0 else None

    def get_host(self):
        boundaries = self._find_boundaries()
        scheme_sep = boundaries['scheme_sep']
        if scheme_sep == -1:
            return None
        
        host_start = scheme_sep + 3
        host_end = len(self.url)
        
        # Find the earliest delimiter after the scheme
        for pos in [boundaries['path_start'], boundaries['query_start'], boundaries['fragment_start']]:
            if pos > host_start:
                host_end = min(host_end, pos)
        
        return self.url[host_start:host_end]

    def get_path(self):
        boundaries = self._find_boundaries()
        path_start = boundaries['path_start']
        if path_start == -1:
            return None
        
        path_end = len(self.url)
        for pos in [boundaries['query_start'], boundaries['fragment_start']]:
            if pos > path_start:
                path_end = min(path_end, pos)
        
        return self.url[path_start:path_end]

    def get_query_params(self):
        boundaries = self._find_boundaries()
        query_start = boundaries['query_start']
        if query_start == -1:
            return None
        
        query_end = boundaries['fragment_start'] if boundaries['fragment_start'] > query_start else len(self.url)
        query_string = self.url[query_start + 1:query_end]
        
        if not query_string:
            return {}
        
        params = {}
        for pair in query_string.split('&'):
            if '=' in pair:
                key, value = pair.split('=', 1)
                params[key] = value
        return params

    def get_fragment(self):
        boundaries = self._find_boundaries()
        fragment_start = boundaries['fragment_start']
        return self.url[fragment_start + 1:] if fragment_start != -1 else None
