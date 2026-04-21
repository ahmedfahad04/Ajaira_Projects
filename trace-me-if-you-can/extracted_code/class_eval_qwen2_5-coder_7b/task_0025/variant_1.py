import json

class CookieManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cookie_data = None

    def update_cookies(self, response):
        self.cookie_data = response['cookies']
        self._write_cookies()

    def fetch_cookies(self):
        try:
            with open(self.file_path, 'r') as file:
                cookies_info = json.load(file)
                return cookies_info
        except FileNotFoundError:
            return {}

    def _write_cookies(self):
        try:
            with open(self.file_path, 'w') as file:
                json.dump(self.cookie_data, file)
            return True
        except Exception as e:
            print(f"Error writing cookies: {e}")
            return False

    def apply_cookies(self, request):
        request['cookies'] = '; '.join([f'{key}={value}' for key, value in self.cookie_data.items()])
