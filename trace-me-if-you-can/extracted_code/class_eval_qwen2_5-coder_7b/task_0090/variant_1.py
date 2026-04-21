class URLParser:
    def __init__(self, input_url):
        self.input_url = input_url

    def extract_protocol(self):
        protocol_end = self.input_url.find("://")
        return self.input_url[:protocol_end] if protocol_end != -1 else None

    def extract_host(self):
        protocol_end = self.input_url.find("://")
        if protocol_end != -1:
            remainder = self.input_url[protocol_end + 3:]
            host_end = remainder.find("/")
            return remainder[:host_end] if host_end != -1 else remainder
        return None

    def extract_path(self):
        protocol_end = self.input_url.find("://")
        if protocol_end != -1:
            remainder = self.input_url[protocol_end + 3:]
            host_end = remainder.find("/")
            return remainder[host_end:] if host_end != -1 else None
        return None

    def parse_query(self):
        query_pos = self.input_url.find("?")
        fragment_pos = self.input_url.find("#")
        if query_pos != -1:
            query_string = self.input_url[query_pos + 1:fragment_pos]
            params = {}
            if query_string:
                pairs = query_string.split("&")
                for pair in pairs:
                    key, value = pair.split("=")
                    params[key] = value
            return params
        return None

    def extract_fragment(self):
        fragment_pos = self.input_url.find("#")
        return self.input_url[fragment_pos + 1:] if fragment_pos != -1 else None
