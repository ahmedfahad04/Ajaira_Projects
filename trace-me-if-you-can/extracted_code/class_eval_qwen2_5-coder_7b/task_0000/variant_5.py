import logging
import datetime


class AccessControlFilter:

    def __init__(self):
        pass

    def intercept(self, request_details):
        request_path = request_details['path']
        request_method = request_details['method']

        if self.path_starts_with(request_path):
            return True

        try:
            jwt_details = self.extract_jwt(request_details)
            user_level = jwt_details['user']['level']
            if user_level > 2:
                self.log_user_activity(jwt_details['user'])
                return True
        except:
            return False

    def path_starts_with(self, path):
        patterns = ["/api", '/login']
        for pattern in patterns:
            if path.startswith(pattern):
                return True
        return False

    def extract_jwt(self, request_details):
        auth_token = request_details['headers']['Authorization']
        user_data = auth_token['user']
        if auth_token['jwt'].startswith(user_data['name']):
            jwt_date_str = auth_token['jwt'].split(user_data['name'])[1]
            jwt_date = datetime.datetime.strptime(jwt_date_str, "%Y-%m-%d")
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):
                return None
        return auth_token

    def log_user_activity(self, user_data):
        user_ip = user_data['address']
        log_message = f"{user_data['name']}{user_ip}{datetime.datetime.now()}"
        logging.log(msg=log_message, level=1)
