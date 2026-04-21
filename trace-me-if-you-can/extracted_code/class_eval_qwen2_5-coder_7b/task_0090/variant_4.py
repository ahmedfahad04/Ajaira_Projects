class URLUtility:
    def __init__(self, url):
        self.url = url

    def get_protocol(self):
        protocol_end = self.url.find("://")
        return self.url[:protocol_end] if protocol_end != -1 else None

    def get_domain(self):
        protocol_end = self.url.find("://")
        if protocol_end != -1:
            remaining = self.url[protocol_end + 3:]
            domain_end = remaining.find("/")
            return remaining[:domain_end] if domain_end != -1 else remaining
        return None

    def get_directory(self):
        protocol_end = self.url.find("://")
        if protocol_end != -1:
            remaining = self.url[protocol_end + 3:]
            domain_end = remaining.find("/")
            return remaining[domain_end:] if domain_end != -1 else None
        return None

    def parse_query_string(self):
        query_start = self.url.find("?")
        fragment_start = self.url.find("#")
        if query_start != -1:
            query_segment = self.url[query_start + 1:fragment_start]
            params = {}
            if query_segment:
                pairs = query_segment.split("&")
                for pair in pairs:
                    key, value = pair.split("=")
                    params[key] = value
            return params
        return None

    def fetch_fragment(self):
        fragment_start = self.url.find("#")
        return self.url[fragment_start + 1:] if fragment_start != -1 else None
