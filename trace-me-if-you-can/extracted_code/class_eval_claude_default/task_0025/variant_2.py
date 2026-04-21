import json

class CookiesUtil:
    def __init__(self, cookies_file):
        self.cookies_file = cookies_file
        self.cookies = None

    def get_cookies(self, reponse):
        self.cookies = reponse['cookies']
        self._save_cookies()

    def load_cookies(self):
        cookies_data = {}
        try:
            with open(self.cookies_file, 'r') as file:
                cookies_data = json.load(file)
        except FileNotFoundError:
            pass
        return cookies_data

    def _save_cookies(self):
        success = True
        try:
            with open(self.cookies_file, 'w') as file:
                json.dump(self.cookies, file)
        except:
            success = False
        return success

    def set_cookies(self, request):
        cookie_pairs = []
        for key, value in self.cookies.items():
            cookie_pairs.append(f'{key}={value}')
        request['cookies'] = '; '.join(cookie_pairs)
