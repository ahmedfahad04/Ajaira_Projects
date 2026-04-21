import json
from contextlib import suppress

class CookiesUtil:
    def __init__(self, cookies_file):
        self.cookies_file = cookies_file
        self.cookies = None

    def get_cookies(self, reponse):
        self.cookies = reponse['cookies']
        self._save_cookies()

    def load_cookies(self):
        with suppress(FileNotFoundError):
            with open(self.cookies_file, 'r') as file:
                return json.load(file)
        return {}

    def _save_cookies(self):
        with suppress(Exception):
            with open(self.cookies_file, 'w') as file:
                json.dump(self.cookies, file)
            return True
        return False

    def set_cookies(self, request):
        cookies_generator = (f'{key}={value}' for key, value in self.cookies.items())
        request['cookies'] = '; '.join(cookies_generator)
