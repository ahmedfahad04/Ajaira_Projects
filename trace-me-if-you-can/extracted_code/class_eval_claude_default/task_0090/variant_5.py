class URLHandler:
    SCHEME_DELIMITER = "://"
    PATH_DELIMITER = "/"
    QUERY_DELIMITER = "?"
    FRAGMENT_DELIMITER = "#"
    PARAM_SEPARATOR = "&"
    KEY_VALUE_SEPARATOR = "="

    def __init__(self, url):
        self.url = url

    def _extract_segment(self, start_delimiter, end_delimiters=None, include_delimiter=False):
        start_pos = self.url.find(start_delimiter)
        if start_pos == -1:
            return None
        
        segment_start = start_pos + len(start_delimiter) if not include_delimiter else start_pos
        segment_end = len(self.url)
        
        if end_delimiters:
            for delimiter in end_delimiters:
                pos = self.url.find(delimiter, segment_start)
                if pos != -1:
                    segment_end = min(segment_end, pos)
        
        return self.url[segment_start:segment_end] if segment_end > segment_start else None

    def get_scheme(self):
        scheme_pos = self.url.find(self.SCHEME_DELIMITER)
        return self.url[:scheme_pos] if scheme_pos != -1 else None

    def get_host(self):
        return self._extract_segment(self.SCHEME_DELIMITER, 
                                   [self.PATH_DELIMITER, self.QUERY_DELIMITER, self.FRAGMENT_DELIMITER])

    def get_path(self):
        scheme_pos = self.url.find(self.SCHEME_DELIMITER)
        if scheme_pos == -1:
            return None
        
        host_section = self.url[scheme_pos + 3:]
        path_start = host_section.find(self.PATH_DELIMITER)
        if path_start == -1:
            return None
        
        full_path_start = scheme_pos + 3 + path_start
        return self._extract_segment("", [self.QUERY_DELIMITER, self.FRAGMENT_DELIMITER])[full_path_start - scheme_pos - 3:]

    def get_query_params(self):
        query_string = self._extract_segment(self.QUERY_DELIMITER, [self.FRAGMENT_DELIMITER])
        if not query_string:
            return None
        
        params = {}
        if query_string:
            for param in query_string.split(self.PARAM_SEPARATOR):
                if self.KEY_VALUE_SEPARATOR in param:
                    key, value = param.split(self.KEY_VALUE_SEPARATOR, 1)
                    params[key] = value
        return params

    def get_fragment(self):
        return self._extract_segment(self.FRAGMENT_DELIMITER)
