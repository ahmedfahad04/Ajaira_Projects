class URLHandler:
    def __init__(self, url):
        self.url = url
        self._scheme_end = self.url.find("://")
        self._query_start = self.url.find("?")
        self._fragment_start = self.url.find("#")

    def get_scheme(self):
        return self.url[:self._scheme_end] if self._scheme_end != -1 else None

    def get_host(self):
        if self._scheme_end == -1:
            return None
        
        start = self._scheme_end + 3
        end = len(self.url)
        
        for delimiter in ["/", "?", "#"]:
            pos = self.url.find(delimiter, start)
            if pos != -1:
                end = min(end, pos)
        
        return self.url[start:end] if end > start else None

    def get_path(self):
        if self._scheme_end == -1:
            return None
            
        path_start = self.url.find("/", self._scheme_end + 3)
        if path_start == -1:
            return None
            
        path_end = len(self.url)
        if self._query_start != -1:
            path_end = min(path_end, self._query_start)
        if self._fragment_start != -1:
            path_end = min(path_end, self._fragment_start)
            
        return self.url[path_start:path_end]

    def get_query_params(self):
        if self._query_start == -1:
            return None
            
        query_end = self._fragment_start if self._fragment_start != -1 else len(self.url)
        query_string = self.url[self._query_start + 1:query_end]
        
        return dict(pair.split("=", 1) for pair in query_string.split("&") 
                   if "=" in pair and query_string) or {}

    def get_fragment(self):
        return self.url[self._fragment_start + 1:] if self._fragment_start != -1 else None
