class URLHandler:
    def __init__(self, url):
        self.url = url
    
    def _split_url(self):
        parts = {'scheme': None, 'authority': None, 'path': None, 'query': None, 'fragment': None}
        
        # Extract fragment
        if '#' in self.url:
            url_part, parts['fragment'] = self.url.rsplit('#', 1)
        else:
            url_part = self.url
            
        # Extract query
        if '?' in url_part:
            url_part, parts['query'] = url_part.rsplit('?', 1)
            
        # Extract scheme and authority
        if '://' in url_part:
            parts['scheme'], remainder = url_part.split('://', 1)
            if '/' in remainder:
                parts['authority'], parts['path'] = remainder.split('/', 1)
                parts['path'] = '/' + parts['path']
            else:
                parts['authority'] = remainder
        
        return parts

    def get_scheme(self):
        return self._split_url()['scheme']

    def get_host(self):
        return self._split_url()['authority']

    def get_path(self):
        return self._split_url()['path']

    def get_query_params(self):
        query = self._split_url()['query']
        if not query:
            return None
        
        params = {}
        for param in query.split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                params[key] = value
        return params

    def get_fragment(self):
        return self._split_url()['fragment']
