import json

class CookieProcessor:
    def __init__(self, cookie_file):
        self.cookie_file = cookie_file
        self.cookie_content = None

    def update_cookie_info(self, response):
        self.cookie_content = response['cookies']
        self._save_cookie_info()

    def fetch_cookie_info(self):
        try:
            with open(self.cookie_file, 'r') as file:
                cookies_data = json.load(file)
                return cookies_data
        except FileNotFoundError:
            return {}

    def _save_cookie_info(self):
        try:
            with open(self.cookie_file, 'w') as file:
                json.dump(self.cookie_content, file)
            return True
        except Exception as e:
            print(f"Error saving cookie info: {e}")
            return False

    def append_cookies_to_request(self, request):
        request['cookies'] = '; '.join([f'{key}={value}' for key, value in self.cookie_content.items()])
