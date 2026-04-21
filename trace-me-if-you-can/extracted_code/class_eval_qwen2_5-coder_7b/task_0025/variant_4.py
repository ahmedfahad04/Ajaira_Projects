import json

class CookieController:
    def __init__(self, path_to_cookies):
        self.path_to_cookies = path_to_cookies
        self.cookie_data = None

    def update_cookie_store(self, response):
        self.cookie_data = response['cookies']
        self._persist_cookies()

    def load_cookie_store(self):
        try:
            with open(self.path_to_cookies, 'r') as file:
                cookies_info = json.load(file)
                return cookies_info
        except FileNotFoundError:
            return {}

    def _persist_cookies(self):
        try:
            with open(self.path_to_cookies, 'w') as file:
                json.dump(self.cookie_data, file)
            return True
        except Exception as e:
            print(f"Error persisting cookies: {e}")
            return False

    def add_cookies_to_request(self, request):
        request['cookies'] = '; '.join([f'{key}={value}' for key, value in self.cookie_data.items()])
