class NumericEntityUnescaper:
    def __init__(self):
        pass

    def replace(self, string):
        chars = list(string)
        write_idx = 0
        read_idx = 0
        
        while read_idx < len(chars):
            if self._is_entity_start(chars, read_idx):
                entity_result = self._process_entity(chars, read_idx)
                if entity_result:
                    replacement_char, next_read_idx = entity_result
                    chars[write_idx] = replacement_char
                    write_idx += 1
                    read_idx = next_read_idx
                else:
                    chars[write_idx] = chars[read_idx]
                    write_idx += 1
                    read_idx += 1
            else:
                chars[write_idx] = chars[read_idx]
                write_idx += 1
                read_idx += 1
        
        return ''.join(chars[:write_idx])
    
    def _is_entity_start(self, chars, idx):
        return (idx + 2 < len(chars) and 
                chars[idx] == '&' and 
                chars[idx + 1] == '#')
    
    def _process_entity(self, chars, start_idx):
        idx = start_idx + 2
        base = 10
        
        if idx < len(chars) and chars[idx].lower() == 'x':
            base = 16
            idx += 1
        
        if idx >= len(chars):
            return None
            
        digit_start = idx
        while idx < len(chars) and self._is_valid_char(chars[idx], base == 16):
            idx += 1
            
        if (idx < len(chars) and 
            chars[idx] == ';' and 
            idx > digit_start):
            try:
                num_str = ''.join(chars[digit_start:idx])
                value = int(num_str, base)
                return chr(value), idx + 1
            except (ValueError, OverflowError):
                return None
        
        return None
    
    @staticmethod
    def _is_valid_char(char, is_hex):
        return char.isdigit() or (is_hex and 'a' <= char.lower() <= 'f')
