class NumericEntityUnescaper:
    def __init__(self):
        pass

    def replace(self, string):
        tokens = self._tokenize(string)
        return ''.join(self._decode_token(token) for token in tokens)
    
    def _tokenize(self, string):
        tokens = []
        i = 0
        
        while i < len(string):
            if i + 2 < len(string) and string[i:i+2] == '&#':
                entity_end = self._find_entity_end(string, i)
                if entity_end != -1:
                    tokens.append(string[i:entity_end+1])
                    i = entity_end + 1
                else:
                    tokens.append(string[i])
                    i += 1
            else:
                tokens.append(string[i])
                i += 1
                
        return tokens
    
    def _find_entity_end(self, string, start):
        i = start + 2
        if i < len(string) and string[i].lower() == 'x':
            i += 1
            
        digit_found = False
        while i < len(string):
            if string[i] == ';':
                return i if digit_found else -1
            elif not self.is_hex_char(string[i]):
                return -1
            digit_found = True
            i += 1
            
        return -1
    
    def _decode_token(self, token):
        if not token.startswith('&#') or not token.endswith(';'):
            return token
            
        content = token[2:-1]
        if not content:
            return token
            
        try:
            if content[0].lower() == 'x':
                return chr(int(content[1:], 16))
            else:
                return chr(int(content, 10))
        except (ValueError, OverflowError):
            return token
    
    @staticmethod
    def is_hex_char(char):
        return char.isdigit() or ('a' <= char.lower() <= 'f')
