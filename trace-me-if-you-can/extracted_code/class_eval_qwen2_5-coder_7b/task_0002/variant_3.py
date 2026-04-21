class ParamInterpreter:
    def __init__(self):
        self.params = {}
        self.required_params = set()
        self.param_types = {}

    def interpret_params(self, command_string):
        params = command_string.split()[1:]
        for i in range(len(params)):
            param = params[i]
            if param.startswith('--'):
                key_value = param[2:].split('=')
                if len(key_value) == 2:
                    self.params[key_value[0]] = self._parse_param(key_value[0], key_value[1])
                else:
                    self.params[key_value[0]] = True
            elif param.startswith('-'):
                key = param[1:]
                if i + 1 < len(params) and not params[i + 1].startswith('-'):
                    self.params[key] = self._parse_param(key, params[i + 1])
                else:
                    self.params[key] = True
        missing_params = self.required_params - set(self.params.keys())
        if missing_params:
            return False, missing_params

        return True, None

    def fetch_param(self, key):
        return self.params.get(key)

    def define_param(self, param, required=False, param_type=str):
        if required:
            self.required_params.add(param)
        self.param_types[param] = param_type

    def _parse_param(self, param, value):
        try:
            return self.param_types[param](value)
        except (ValueError, KeyError):
            return value
