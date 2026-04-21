class URLAnalysis:
    def __init__(self, url):
        self.url = url

    def fetch_protocol(self):
        protocol_index = self.url.find("://")
        return self.url[:protocol_index] if protocol_index != -1 else None

    def fetch_host(self):
        protocol_index = self.url.find("://")
        if protocol_index != -1:
            remaining = self.url[protocol_index + 3:]
            host_index = remaining.find("/")
            return remaining[:host_index] if host_index != -1 else remaining
        return None

    def fetch_path(self):
        protocol_index = self.url.find("://")
        if protocol_index != -1:
            remaining = self.url[protocol_index + 3:]
            host_index = remaining.find("/")
            return remaining[host_index:] if host_index != -1 else None
        return None

    def dissect_query(self):
        query_index = self.url.find("?")
        fragment_index = self.url.find("#")
        if query_index != -1:
            query_segment = self.url[query_index + 1:fragment_index]
            params = {}
            if query_segment:
                pairs = query_segment.split("&")
                for pair in pairs:
                    key, value = pair.split("=")
                    params[key] = value
            return params
        return None

    def extract_fragment(self):
        fragment_index = self.url.find("#")
        return self.url[fragment_index + 1:] if fragment_index != -1 else None
