import json

class CookieHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.cookie_store = None

    def receive_cookies(self, response):
        self.cookie_store = response['cookies']
        self._store_cookies()

    def retrieve_cookies(self):
        try:
            with open(self.filepath, 'r') as file:
                cookies_content = json.load(file)
                return cookies_content
        except FileNotFoundError:
            return {}

    def _store_cookies(self):
        try:
            with open(self.filepath, 'w') as file:
                json.dump(self.cookie_store, file)
            return True
        except Exception as err:
            print(f"Error storing cookies: {err}")
            return False

    def incorporate_cookies(self, request):
        request['cookies'] = '; '.join([f'{key}={value}' for key, value in self.cookie_store.items()])
