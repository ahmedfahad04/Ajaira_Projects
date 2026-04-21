class NumericEntityUnescaper:
    def __init__(self):
        pass

    def replace(self, string):
        result = []
        i = 0
        
        while i < len(string):
            entity_info = self._try_parse_entity(string, i)
            if entity_info:
                char_code, new_pos = entity_info
                try:
                    result.append(chr(char_code))
                    i = new_pos
                except (ValueError, OverflowError):
                    result.append(string[i])
                    i += 1
            else:
                result.append(string[i])
                i += 1
                
        return ''.join(result)
    
    def _try_parse_entity(self, string, pos):
        if pos + 2 >= len(string) or string[pos:pos+2] != '&#':
            return None
            
        start_pos = pos + 2
        is_hex = start_pos < len(string) and string[start_pos].lower() == 'x'
        
        if is_hex:
            start_pos += 1
            
        if start_pos >= len(string):
            return None
            
        end_pos = start_pos
        while end_pos < len(string) and self._is_valid_digit(string[end_pos], is_hex):
            end_pos += 1
            
        if end_pos < len(string) and string[end_pos] == ';' and end_pos > start_pos:
            number_str = string[start_pos:end_pos]
            base = 16 if is_hex else 10
            try:
                return int(number_str, base), end_pos + 1
            except ValueError:
                return None
        
        return None
    
    def _is_valid_digit(self, char, is_hex):
        if is_hex:
            return char.isdigit() or ('a' <= char.lower() <= 'f')
        return char.isdigit()
