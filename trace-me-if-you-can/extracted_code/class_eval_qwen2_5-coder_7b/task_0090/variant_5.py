class URLDataExtractor:
    def __init__(self, url):
        self.url = url

    def extract_protocol(self):
        protocol_end = self.url.find("://")
        return self.url[:protocol_end] if protocol_end != -1 else None

    def extract_server(self):
        protocol_end = self.url.find("://")
        if protocol_end != -1:
            server_end = self.url.find("/", protocol_end + 3)
            return self.url[protocol_end + 3:server_end] if server_end != -1 else self.url[protocol_end + 3:]
        return None

    def extract_directory(self):
        protocol_end = self.url.find("://")
        if protocol_end != -1:
            server_end = self.url.find("/", protocol_end + 3)
            path_start = self.url.find("/", server_end)
            return self.url[path_start:] if path_start != -1 else "/"
        return None

    def parse_query(self):
        query_start = self.url.find("?")
        fragment_start = self.url.find("#")
        if query_start != -1:
            query_string = self.url[query_start + 1:fragment_start]
            params = {}
            if query_string:
                pairs = query_string.split("&")
                for pair in pairs:
                    key, value = pair.split("=")
                    params[key] = value
            return params
        return None

    def extract_hash(self):
        fragment_start = self.url.find("#")
        return self.url[fragment_start + 1:] if fragment_start != -1 else None
