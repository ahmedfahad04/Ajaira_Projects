import logging
import datetime


class RequestFilter:

    def __init__(self):
        pass

    def process_request(self, request_data):
        uri = request_data['path']
        method = request_data['method']

        if self.uri_starts_with(uri):
            return True

        try:
            auth_token = self.extract_jwt_token(request_data)
            user_info = auth_token['user']
            if user_info['level'] > 2:
                self.log_user_details(user_info)
                return True
        except:
            return False

    def uri_starts_with(self, uri):
        start_patterns = ["/api", '/login']
        for pattern in start_patterns:
            if uri.startswith(pattern):
                return True
        return False

    def extract_jwt_token(self, request_data):
        token = request_data['headers']['Authorization']
        user_details = token['user']
        if token['jwt'].startswith(user_details['name']):
            jwt_date_str = token['jwt'].split(user_details['name'])[1]
            jwt_date = datetime.datetime.strptime(jwt_date_str, "%Y-%m-%d")
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):
                return None
        return token

    def log_user_details(self, user_details):
        host_address = user_details['address']
        log_message = f"{user_details['name']}{host_address}{datetime.datetime.now()}"
        logging.log(msg=log_message, level=1)
