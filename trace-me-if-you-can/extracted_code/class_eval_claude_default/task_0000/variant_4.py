import logging
import datetime


class AccessGatewayFilter:

    def __init__(self):
        self.filter_chain = [
            self._handle_public_endpoints,
            self._handle_authenticated_endpoints
        ]

    def filter(self, request):
        context = {'request': request, 'authorized': False}
        
        for handler in self.filter_chain:
            handler(context)
            if context['authorized']:
                return True
        
        return False

    def _handle_public_endpoints(self, context):
        request_uri = context['request']['path']
        if self.is_start_with(request_uri):
            context['authorized'] = True

    def _handle_authenticated_endpoints(self, context):
        if context['authorized']:
            return
            
        request = context['request']
        user_info = self._get_user_info_safely(request)
        
        if user_info and self._check_user_authorization(user_info):
            self.set_current_user_info_and_log(user_info)
            context['authorized'] = True

    def _get_user_info_safely(self, request):
        try:
            token = self.get_jwt_user(request)
            return token['user'] if token else None
        except:
            return None

    def _check_user_authorization(self, user):
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
