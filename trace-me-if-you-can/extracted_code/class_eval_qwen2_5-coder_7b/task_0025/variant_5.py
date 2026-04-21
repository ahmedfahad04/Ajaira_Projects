import json

class CookieService:
    def __init__(self, cookies_path):
        self.cookies_path = cookies_path
        self.cookies = None

    def updateCookies(self, response):
        self.cookies = response['cookies']
        self._saveCookies()

    def loadCookies(self):
        try:
            with open(self.cookies_path, 'r') as file:
                cookies_data = json.load(file)
                return cookies_data
        except FileNotFoundError:
            return {}

    def _saveCookies(self):
        try:
            with open(self.cookies_path, 'w') as file:
                json.dump(self.cookies, file)
            return True
        except Exception as err:
            print(f"Error saving cookies: {err}")
            return False

    def appendCookies(self, request):
        request['cookies'] = '; '.join([f'{key}={value}' for key, value in self.cookies.items()])
