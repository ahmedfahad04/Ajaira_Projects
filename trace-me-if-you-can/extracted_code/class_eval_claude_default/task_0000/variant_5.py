import logging
import datetime


class AccessGatewayFilter:

    def __init__(self):
        pass

    def filter(self, request):
        return (self._try_public_access(request) or 
                self._try_authenticated_access(request))

    def _try_public_access(self, request):
        return self.is_start_with(request['path'])

    def _try_authenticated_access(self, request):
        def safe_get_user():
            try:
                token = self.get_jwt_user(request)
                return token['user'] if token else None
            except:
                return None

        def is_high_level_user(user):
            return user and user['level'] > 2

        def authorize_and_log(user):
            self.set_current_user_info_and_log(user)
            return True

        user = safe_get_user()
        return authorize_and_log(user) if is_high_level_user(user) else False

    def is_start_with(self, request_uri):
        allowed_prefixes = ["/api", '/login']
        return next((True for prefix in allowed_prefixes 
                    if request_uri.startswith(prefix)), False)

    def get_jwt_user(self, request):
        token = request['headers']['Authorization']
        user = token['user']
        
        if not token['jwt'].startswith(user['name']):
            return token
            
        jwt_str_date = token['jwt'].split(user['name'])[1]
        jwt_date = datetime.datetime.strptime(jwt_str_date, "%Y-%m-%d")
        
        days_elapsed = datetime.datetime.today() - jwt_date
        return None if days_elapsed >= datetime.timedelta(days=3) else token

    def set_current_user_info_and_log(self, user):
        timestamp = str(datetime.datetime.now())
        log_message = f"{user['name']}{user['address']}{timestamp}"
        logging.log(msg=log_message, level=1)
