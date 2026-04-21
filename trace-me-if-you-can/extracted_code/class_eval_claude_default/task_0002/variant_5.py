class ArgumentParser:
    def __init__(self):
        self.arguments = {}
        self.required = set()
        self.types = {}
        
        # Strategy pattern for different argument types
        self.parsing_strategies = {
            'long_with_value': self._parse_long_with_value,
            'long_flag': self._parse_long_flag,
            'short_with_value': self._parse_short_with_value,
            'short_flag': self._parse_short_flag
        }

    def parse_arguments(self, command_string):
        tokens = command_string.split()[1:]
        token_queue = list(reversed(tokens))  # Use as stack
        
        while token_queue:
            current_token = token_queue.pop()
            strategy_key = self._determine_strategy(current_token, token_queue)
            
            if strategy_key:
                consumed_extra = self.parsing_strategies[strategy_key](current_token, token_queue)
                # Remove consumed tokens from queue
                for _ in range(consumed_extra):
                    if token_queue:
                        token_queue.pop()
        
        return self._check_completion()

    def _determine_strategy(self, token, remaining_queue):
        if token.startswith('--'):
            return 'long_with_value' if '=' in token else 'long_flag'
        elif token.startswith('-'):
            has_next = remaining_queue and not remaining_queue[-1].startswith('-')
            return 'short_with_value' if has_next else 'short_flag'
        return None

    def _parse_long_with_value(self, token, queue):
        parts = token[2:].split('=', 1)
        key, value = parts[0], parts[1] if len(parts) > 1 else 'True'
        self.arguments[key] = self._convert_type(key, value) if len(parts) > 1 else True
        return 0

    def _parse_long_flag(self, token, queue):
        key = token[2:]
        self.arguments[key] = True
        return 0

    def _parse_short_with_value(self, token, queue):
        key = token[1:]
        value = queue[-1]  # Peek at next token
        self.arguments[key] = self._convert_type(key, value)
        return 1  # Consume the value token

    def _parse_short_flag(self, token, queue):
        key = token[1:]
        self.arguments[key] = True
        return 0

    def _check_completion(self):
        missing_required = self.required - set(self.arguments.keys())
        return (False, missing_required) if missing_required else (True, None)

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
