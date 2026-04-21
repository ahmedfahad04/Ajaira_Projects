class CommandHandler:
    def __init__(self):
        self.commands = {}
        self.required_commands = set()
        self.command_types = {}

    def handle_commands(self, command_string):
        args = command_string.split()[1:]
        for i in range(len(args)):
            arg = args[i]
            if arg.startswith('--'):
                key_value = arg[2:].split('=')
                if len(key_value) == 2:
                    self.commands[key_value[0]] = self._parse_command(key_value[0], key_value[1])
                else:
                    self.commands[key_value[0]] = True
            elif arg.startswith('-'):
                key = arg[1:]
                if i + 1 < len(args) and not args[i + 1].startswith('-'):
                    self.commands[key] = self._parse_command(key, args[i + 1])
                else:
                    self.commands[key] = True
        missing_commands = self.required_commands - set(self.commands.keys())
        if missing_commands:
            return False, missing_commands

        return True, None

    def get_command(self, key):
        return self.commands.get(key)

    def add_command(self, cmd, required=False, cmd_type=str):
        if required:
            self.required_commands.add(cmd)
        self.command_types[cmd] = cmd_type

    def _parse_command(self, cmd, value):
        try:
            return self.command_types[cmd](value)
        except (ValueError, KeyError):
            return value
