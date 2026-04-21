class CmdParser:
    def __init__(self):
        self.cmd_args = {}
        self.required_args = set()
        self.value_types = {}

    def parse_cmd_args(self, command_string):
        args_list = command_string.split()[1:]
        for i in range(len(args_list)):
            arg = args_list[i]
            if arg.startswith('--'):
                key_value = arg[2:].split('=')
                if len(key_value) == 2:
                    self.cmd_args[key_value[0]] = self._parse_value(key_value[0], key_value[1])
                else:
                    self.cmd_args[key_value[0]] = True
            elif arg.startswith('-'):
                key = arg[1:]
                if i + 1 < len(args_list) and not args_list[i + 1].startswith('-'):
                    self.cmd_args[key] = self._parse_value(key, args_list[i + 1])
                else:
                    self.cmd_args[key] = True
        missing_args = self.required_args - set(self.cmd_args.keys())
        if missing_args:
            return False, missing_args

        return True, None

    def get_cmd_arg(self, key):
        return self.cmd_args.get(key)

    def add_cmd_arg(self, arg, required=False, arg_type=str):
        if required:
            self.required_args.add(arg)
        self.value_types[arg] = arg_type

    def _parse_value(self, arg, value):
        try:
            return self.value_types[arg](value)
        except (ValueError, KeyError):
            return value
