import json
import os

class CookiesUtil:
    def __init__(self, cookies_file):
        self.cookies_file = cookies_file
        self.cookies = None

    def get_cookies(self, reponse):
        self.cookies = reponse['cookies']
        self._save_cookies()

    def load_cookies(self):
        return self._read_json_file() if os.path.exists(self.cookies_file) else {}

    def _read_json_file(self):
        with open(self.cookies_file, 'r') as file:
            return json.load(file)

    def _save_cookies(self):
        return self._write_json_file(self.cookies)

    def _write_json_file(self, data):
        try:
            with open(self.cookies_file, 'w') as file:
                json.dump(data, file)
            return True
        except:
            return False

    def set_cookies(self, request):
        cookie_list = [f'{key}={value}' for key, value in self.cookies.items()]
        request['cookies'] = '; '.join(cookie_list)
