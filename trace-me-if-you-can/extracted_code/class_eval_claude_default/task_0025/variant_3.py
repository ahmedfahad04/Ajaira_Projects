import json

class CookiesUtil:
    def __init__(self, cookies_file):
        self.cookies_file = cookies_file
        self.cookies = None

    def get_cookies(self, reponse):
        self.cookies = reponse.get('cookies')
        self._save_cookies()

    def load_cookies(self):
        def _read_file():
            with open(self.cookies_file, 'r') as file:
                return json.load(file)
        
        try:
            return _read_file()
        except FileNotFoundError:
            return {}

    def _save_cookies(self):
        def _write_file():
            with open(self.cookies_file, 'w') as file:
                json.dump(self.cookies, file)
                return True
        
        try:
            return _write_file()
        except:
            return False

    def set_cookies(self, request):
        def _format_cookie(item):
            key, value = item
            return f'{key}={value}'
        
        formatted_cookies = map(_format_cookie, self.cookies.items())
        request['cookies'] = '; '.join(formatted_cookies)
