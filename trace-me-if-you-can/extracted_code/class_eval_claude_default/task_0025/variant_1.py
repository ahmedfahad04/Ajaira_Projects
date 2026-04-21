import json
from pathlib import Path

class CookiesUtil:
    def __init__(self, cookies_file):
        self.cookies_file = Path(cookies_file)
        self.cookies = None

    def get_cookies(self, reponse):
        self.cookies = reponse['cookies']
        self._save_cookies()

    def load_cookies(self):
        if not self.cookies_file.exists():
            return {}
        
        with open(self.cookies_file, 'r') as file:
            return json.load(file)

    def _save_cookies(self):
        try:
            self.cookies_file.write_text(json.dumps(self.cookies))
            return True
        except:
            return False

    def set_cookies(self, request):
        cookie_string = '; '.join(f'{key}={value}' for key, value in self.cookies.items())
        request['cookies'] = cookie_string
