class CommandLineParser:
    def __init__(self):
        self.args = {}
        self.required_args = set()
        self.arg_types = {}

    def parse_command_line(self, command_string):
        params = command_string.split()[1:]
        for i in range(len(params)):
            param = params[i]
            if param.startswith('--'):
                key_value = param[2:].split('=')
                if len(key_value) == 2:
                    self.args[key_value[0]] = self._parse_arg(key_value[0], key_value[1])
                else:
                    self.args[key_value[0]] = True
            elif param.startswith('-'):
                key = param[1:]
                if i + 1 < len(params) and not params[i + 1].startswith('-'):
                    self.args[key] = self._parse_arg(key, params[i + 1])
                else:
                    self.args[key] = True
        missing_args = self.required_args - set(self.args.keys())
        if missing_args:
            return False, missing_args

        return True, None

    def get_arg(self, key):
        return self.args.get(key)

    def define_arg(self, arg, required=False, arg_type=str):
        if required:
            self.required_args.add(arg)
        self.arg_types[arg] = arg_type

    def _parse_arg(self, arg, value):
        try:
            return self.arg_types[arg](value)
        except (ValueError, KeyError):
            return value
