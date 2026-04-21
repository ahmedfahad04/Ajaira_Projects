import urllib.parse

class URLInfo:
    def __init__(self, url):
        self.url = url

    def get_protocol(self):
        parsed_url = urllib.parse.urlparse(self.url)
        return parsed_url.scheme

    def get_hostname(self):
        parsed_url = urllib.parse.urlparse(self.url)
        return parsed_url.hostname

    def get_path(self):
        parsed_url = urllib.parse.urlparse(self.url)
        return parsed_url.path

    def get_query_params(self):
        parsed_url = urllib.parse.urlparse(self.url)
        return urllib.parse.parse_qs(parsed_url.query)

    def get_fragment(self):
        parsed_url = urllib.parse.urlparse(self.url)
        return parsed_url.fragment
