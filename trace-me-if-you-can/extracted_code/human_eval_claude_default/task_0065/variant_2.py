s = str(x)
string_length = len(s)
effective_shift = min(shift, string_length)
if effective_shift == string_length:
    return s[::-1]
left_part = s[string_length - effective_shift:]
right_part = s[:string_length - effective_shift]
return left_part + right_part
