import logging
import datetime


class RequestValidator:

    def __init__(self):
        pass

    def authorize_request(self, req_info):
        request_uri = req_info['path']
        request_method = req_info['method']

        if self.uri_begins_with(request_uri):
            return True

        try:
            jwt_data = self.decode_jwt(req_info)
            user_level = jwt_data['user']['level']
            if user_level > 2:
                self.log_user_access(jwt_data['user'])
                return True
        except:
            return False

    def uri_begins_with(self, uri):
        start_patterns = ["/api", '/login']
        for pattern in start_patterns:
            if uri.startswith(pattern):
                return True
        return False

    def decode_jwt(self, req_info):
        auth_token = req_info['headers']['Authorization']
        user_info = auth_token['user']
        if auth_token['jwt'].startswith(user_info['name']):
            jwt_date_str = auth_token['jwt'].split(user_info['name'])[1]
            jwt_date = datetime.datetime.strptime(jwt_date_str, "%Y-%m-%d")
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):
                return None
        return auth_token

    def log_user_access(self, user_info):
        user_ip = user_info['address']
        log_entry = f"{user_info['name']}{user_ip}{datetime.datetime.now()}"
        logging.log(msg=log_entry, level=1)
