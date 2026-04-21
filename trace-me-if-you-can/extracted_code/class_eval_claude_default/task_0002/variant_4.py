class ArgumentParser:
    def __init__(self):
        self.arguments = {}
        self.required = set()
        self.types = {}

    def parse_arguments(self, command_string):
        tokens = command_string.split()[1:]
        
        # Functional approach using reduce-like pattern
        def process_tokens(acc, token_with_index):
            token, index = token_with_index
            remaining_tokens = tokens[index + 1:]
            
            processors = [
                (lambda t: t.startswith('--'), self._process_double_dash),
                (lambda t: t.startswith('-'), self._process_single_dash),
                (lambda t: True, lambda t, r, a: (a, 0))  # default: do nothing
            ]
            
            for condition, processor in processors:
                if condition(token):
                    updated_args, skip_count = processor(token, remaining_tokens, acc)
                    return updated_args, skip_count
            
            return acc, 0

        # Process all tokens with their indices
        arguments = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            arguments, skip = process_tokens(arguments, (token, i))[0], process_tokens(arguments, (token, i))[1]
            i += 1 + skip

        self.arguments = arguments
        missing_args = self.required - set(self.arguments.keys())
        return (True, None) if not missing_args else (False, missing_args)

    def _process_double_dash(self, token, remaining_tokens, current_args):
        key_value_pair = token[2:].split('=', 1)
        key = key_value_pair[0]
        
        if len(key_value_pair) == 2:
            current_args[key] = self._convert_type(key, key_value_pair[1])
        else:
            current_args[key] = True
        
        return current_args, 0

    def _process_single_dash(self, token, remaining_tokens, current_args):
        key = token[1:]
        
        has_value = (remaining_tokens and 
                    not remaining_tokens[0].startswith('-'))
        
        if has_value:
            current_args[key] = self._convert_type(key, remaining_tokens[0])
            return current_args, 1
        else:
            current_args[key] = True
            return current_args, 0

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
