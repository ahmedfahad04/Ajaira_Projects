import logging
import datetime
from functools import wraps


def exception_handler(default_return=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                return default_return
        return wrapper
    return decorator


class AccessGatewayFilter:

    def __init__(self):
        self._public_paths = {"/api", '/login'}

    def filter(self, request):
        request_uri = request['path']
        
        # Check public endpoints first
        if self._check_public_access(request_uri):
            return True
        
        # Validate user authentication and authorization
        return self._validate_authenticated_access(request)

    def _check_public_access(self, request_uri):
        return any(request_uri.startswith(path) for path in self._public_paths)

    @exception_handler(default_return=False)
    def _validate_authenticated_access(self, request):
        token = self.get_jwt_user(request)
        user = token['user']
        
        if self._is_authorized_user(user):
            self.set_current_user_info_and_log(user)
            return True
        return False

    def _is_authorized_user(self, user):
        return user['level'] > 2

    def is_start_with(self, request_uri):
        start_with = ["/api", '/login']
        for s in start_with:
            if request_uri.startswith(s):
                return True
        return False

    def get_jwt_user(self, request):
        token = request['headers']['Authorization']
        user = token['user']
        if token['jwt'].startswith(user['name']):
            jwt_str_date = token['jwt'].split(user['name'])[1]
            jwt_date = datetime.datetime.strptime(jwt_str_date, "%Y-%m-%d")
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):
                return None
        return token

    def set_current_user_info_and_log(self, user):
        host = user['address']
        logging.log(msg=user['name'] + host + str(datetime.datetime.now()), level=1)
