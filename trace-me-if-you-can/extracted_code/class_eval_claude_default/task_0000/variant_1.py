import logging
import datetime
from typing import Dict, Any, Optional


class AccessGatewayFilter:
    ALLOWED_PREFIXES = ["/api", '/login']
    MIN_USER_LEVEL = 3
    TOKEN_EXPIRY_DAYS = 3

    def __init__(self):
        pass

    def filter(self, request: Dict[str, Any]) -> bool:
        if self._is_public_endpoint(request['path']):
            return True
        
        token = self._extract_and_validate_token(request)
        if not token:
            return False
            
        user = token['user']
        if user['level'] >= self.MIN_USER_LEVEL:
            self._log_user_access(user)
            return True
        
        return False

    def _is_public_endpoint(self, request_uri: str) -> bool:
        return any(request_uri.startswith(prefix) for prefix in self.ALLOWED_PREFIXES)

    def _extract_and_validate_token(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return self.get_jwt_user(request)
        except:
            return None

    def get_jwt_user(self, request):
        token = request['headers']['Authorization']
        user = token['user']
        if token['jwt'].startswith(user['name']):
            jwt_str_date = token['jwt'].split(user['name'])[1]
            jwt_date = datetime.datetime.strptime(jwt_str_date, "%Y-%m-%d")
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=self.TOKEN_EXPIRY_DAYS):
                return None
        return token

    def _log_user_access(self, user: Dict[str, Any]) -> None:
        self.set_current_user_info_and_log(user)

    def set_current_user_info_and_log(self, user):
        host = user['address']
        logging.log(msg=user['name'] + host + str(datetime.datetime.now()), level=1)
