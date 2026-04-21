class OptionParser:
    def __init__(self):
        self.options = {}
        self.mandatory = set()
        self.data_types = {}

    def parse_options(self, command_string):
        params = command_string.split()[1:]
        for i in range(len(params)):
            param = params[i]
            if param.startswith('--'):
                key_value = param[2:].split('=')
                if len(key_value) == 2:
                    self.options[key_value[0]] = self._type_conversion(key_value[0], key_value[1])
                else:
                    self.options[key_value[0]] = True
            elif param.startswith('-'):
                key = param[1:]
                if i + 1 < len(params) and not params[i + 1].startswith('-'):
                    self.options[key] = self._type_conversion(key, params[i + 1])
                else:
                    self.options[key] = True
        missing_keys = self.mandatory - set(self.options.keys())
        if missing_keys:
            return False, missing_keys

        return True, None

    def fetch_option(self, key):
        return self.options.get(key)

    def define_option(self, opt, mandatory=False, opt_type=str):
        if mandatory:
            self.mandatory.add(opt)
        self.data_types[opt] = opt_type

    def _type_conversion(self, opt, value):
        try:
            return self.data_types[opt](value)
        except (ValueError, KeyError):
            return value
