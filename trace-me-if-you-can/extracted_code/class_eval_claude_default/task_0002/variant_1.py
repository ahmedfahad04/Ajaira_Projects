class ArgumentParser:
    def __init__(self):
        self.arguments = {}
        self.required = set()
        self.types = {}

    def parse_arguments(self, command_string):
        tokens = command_string.split()[1:]
        iterator = iter(enumerate(tokens))
        
        for i, token in iterator:
            if token.startswith('--'):
                self._handle_long_option(token)
            elif token.startswith('-'):
                remaining_tokens = tokens[i+1:] if i+1 < len(tokens) else []
                consumed = self._handle_short_option(token, remaining_tokens)
                # Skip the next token if it was consumed as a value
                if consumed:
                    try:
                        next(iterator)
                    except StopIteration:
                        pass
        
        missing_args = self.required - set(self.arguments.keys())
        return (False, missing_args) if missing_args else (True, None)

    def _handle_long_option(self, token):
        parts = token[2:].split('=', 1)
        key = parts[0]
        value = self._convert_type(key, parts[1]) if len(parts) == 2 else True
        self.arguments[key] = value

    def _handle_short_option(self, token, remaining_tokens):
        key = token[1:]
        if remaining_tokens and not remaining_tokens[0].startswith('-'):
            self.arguments[key] = self._convert_type(key, remaining_tokens[0])
            return True
        else:
            self.arguments[key] = True
            return False

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
