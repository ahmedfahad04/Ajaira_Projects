class NumericEntityUnescaper:
    def __init__(self):
        self.hex_chars = set('0123456789abcdefABCDEF')
        self.decimal_chars = set('0123456789')

    def replace(self, string):
        return self._scan_and_replace(string, 0, [])
    
    def _scan_and_replace(self, string, index, accumulator):
        if index >= len(string):
            return ''.join(accumulator)
            
        if self._can_start_entity(string, index):
            entity_data = self._extract_entity(string, index)
            if entity_data:
                char_value, next_index = entity_data
                accumulator.append(chr(char_value))
                return self._scan_and_replace(string, next_index, accumulator)
        
        accumulator.append(string[index])
        return self._scan_and_replace(string, index + 1, accumulator)
    
    def _can_start_entity(self, string, index):
        return (index + 2 < len(string) and 
                string[index] == '&' and 
                string[index + 1] == '#')
    
    def _extract_entity(self, string, start_index):
        cursor = start_index + 2
        radix = 10
        valid_chars = self.decimal_chars
        
        if cursor < len(string) and string[cursor].lower() == 'x':
            radix = 16
            valid_chars = self.hex_chars
            cursor += 1
        
        if cursor >= len(string):
            return None
            
        number_start = cursor
        while cursor < len(string) and string[cursor] in valid_chars:
            cursor += 1
            
        if (cursor < len(string) and 
            string[cursor] == ';' and 
            cursor > number_start):
            try:
                digits = string[number_start:cursor]
                numeric_value = int(digits, radix)
                return numeric_value, cursor + 1
            except ValueError:
                return None
        
        return None
