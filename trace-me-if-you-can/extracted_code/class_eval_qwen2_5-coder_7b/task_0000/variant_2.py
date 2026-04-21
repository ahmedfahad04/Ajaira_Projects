import logging
import datetime


class ApiAccessControl:

    def __init__(self):
        pass

    def validate_request(self, req):
        endpoint = req['path']
        method = req['method']

        if self.path_matches(endpoint):
            return True

        try:
            jwt_token = self.extract_user_token(req)
            user_level = jwt_token['user']['level']
            if user_level > 2:
                self.record_user_activity(jwt_token['user'])
                return True
        except:
            return False

    def path_matches(self, endpoint):
        patterns = ["/api", '/login']
        for pattern in patterns:
            if endpoint.startswith(pattern):
                return True
        return False

    def extract_user_token(self, req):
        auth_header = req['headers']['Authorization']
        user_info = auth_header['user']
        if auth_header['jwt'].startswith(user_info['name']):
            jwt_date_part = auth_header['jwt'].split(user_info['name'])[1]
            jwt_date = datetime.datetime.strptime(jwt_date_part, "%Y-%m-%d")
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):
                return None
        return auth_header

    def record_user_activity(self, user_info):
        client_ip = user_info['address']
        log_entry = f"{user_info['name']}{client_ip}{datetime.datetime.now()}"
        logging.log(msg=log_entry, level=1)
