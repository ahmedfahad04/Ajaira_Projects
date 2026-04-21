class ArgumentParser:
    def __init__(self):
        self.arguments = {}
        self.required = set()
        self.types = {}

    def parse_arguments(self, command_string):
        args = command_string.split()[1:]
        
        # Build argument parsing state machine
        parsing_states = {
            'SCAN': self._scan_token,
            'EXPECT_VALUE': self._expect_value,
            'COMPLETE': lambda *args: None
        }
        
        state = 'SCAN'
        context = {'pending_key': None, 'index': 0}
        
        while context['index'] < len(args) and state != 'COMPLETE':
            current_arg = args[context['index']]
            state = parsing_states[state](current_arg, args, context)
            if state != 'EXPECT_VALUE':
                context['index'] += 1
        
        # Handle any pending state
        if state == 'EXPECT_VALUE':
            self.arguments[context['pending_key']] = True
        
        missing_args = self.required - set(self.arguments.keys())
        return (True, None) if not missing_args else (False, missing_args)

    def _scan_token(self, token, args, context):
        if token.startswith('--'):
            return self._process_long_option(token, context)
        elif token.startswith('-'):
            return self._process_short_option(token, args, context)
        return 'SCAN'

    def _process_long_option(self, token, context):
        key_value = token[2:].split('=', 1)
        if len(key_value) == 2:
            self.arguments[key_value[0]] = self._convert_type(key_value[0], key_value[1])
            return 'SCAN'
        else:
            self.arguments[key_value[0]] = True
            return 'SCAN'

    def _process_short_option(self, token, args, context):
        key = token[1:]
        next_index = context['index'] + 1
        if next_index < len(args) and not args[next_index].startswith('-'):
            context['pending_key'] = key
            return 'EXPECT_VALUE'
        else:
            self.arguments[key] = True
            return 'SCAN'

    def _expect_value(self, token, args, context):
        key = context['pending_key']
        self.arguments[key] = self._convert_type(key, token)
        context['pending_key'] = None
        context['index'] += 1
        return 'SCAN'

    def get_argument(self, key):
        return self.arguments.get(key)

    def add_argument(self, arg, required=False, arg_type=str):
        if required:
            self.required.add(arg)
        self.types[arg] = arg_type

    def _convert_type(self, arg, value):
        try:
            return self.types[arg](value)
        except (ValueError, KeyError):
            return value
