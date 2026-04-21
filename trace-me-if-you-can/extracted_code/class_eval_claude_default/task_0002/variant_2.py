import re
from typing import Dict, Set, Callable, Any, Tuple, Optional, Union

class ArgumentParser:
    def __init__(self):
        self.arguments: Dict[str, Any] = {}
        self.required: Set[str] = set()
        self.types: Dict[str, Callable] = {}

    def parse_arguments(self, command_string: str) -> Tuple[bool, Optional[Set[str]]]:
        tokens = self._tokenize(command_string)
        parsers = [
            (r'^--([^=]+)=(.+)$', self._parse_long_with_value),
            (r'^--(.+)$', self._parse_long_flag),
            (r'^-(.+)$', self._parse_short_option)
        ]
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            for pattern, parser in parsers:
                match = re.match(pattern, token)
                if match:
                    i += parser(match, tokens, i)
                    break
            else:
                i += 1
        
        return self._validate_required_args()

    def _tokenize(self, command_string: str) -> list:
        return command_string.split()[1:]

    def _parse_long_with_value(self, match, tokens: list, index: int) -> int:
        key, value = match.groups()
        self.arguments[key] = self._convert_type(key, value)
        return 1

    def _parse_long_flag(self, match, tokens: list, index: int) -> int:
        key = match.group(1)
        self.arguments[key] = True
        return 1

    def _parse_short_option(self, match, tokens: list, index: int) -> int:
        key = match.group(1)
        if index + 1 < len(tokens) and not tokens[index + 1].startswith('-'):
            self.arguments[key] = self._convert_type(key, tokens[index + 1])
            return 2
        else:
            self.arguments[key] = True
            return 1

    def _validate_required_args(self) -> Tuple[bool, Optional[Set[str]]]:
        missing = self.required - set(self.arguments.keys())
        return (False, missing) if missing else (True, None)

    def get_argument(self, key: str) -> Any:
        return self.arguments.get(key)

    def add_argument(self, arg: str, required: bool = False, arg_type: Callable = str):
        if required:
            self.required.add(arg)
        self.types[arg] = arg_type

    def _convert_type(self, arg: str, value: str) -> Any:
        try:
            return self.types[arg](value)
        except (ValueError, KeyError):
            return value
