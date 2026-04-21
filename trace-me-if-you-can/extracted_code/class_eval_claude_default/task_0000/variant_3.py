import logging
import datetime


class AuthenticationStrategy:
    def authenticate(self, request):
        raise NotImplementedError

class PublicEndpointStrategy(AuthenticationStrategy):
    def __init__(self, allowed_prefixes):
        self.allowed_prefixes = allowed_prefixes
    
    def authenticate(self, request):
        request_uri = request['path']
        return any(request_uri.startswith(prefix) for prefix in self.allowed_prefixes)

class JWTAuthenticationStrategy(AuthenticationStrategy):
    def __init__(self, gateway_filter):
        self.gateway_filter = gateway_filter
    
    def authenticate(self, request):
        try:
            token = self.gateway_filter.get_jwt_user(request)
            user = token['user']
            if user['level'] > 2:
                self.gateway_filter.set_current_user_info_and_log(user)
                return True
        except:
            pass
        return False


class AccessGatewayFilter:

    def __init__(self):
        self.public_strategy = PublicEndpointStrategy(["/api", '/login'])
        self.jwt_strategy = JWTAuthenticationStrategy(self)

    def filter(self, request):
        # Try public endpoint authentication first
        if self.public_strategy.authenticate(request):
            return True
        
        # Fall back to JWT authentication
        return self.jwt_strategy.authenticate(request)

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
