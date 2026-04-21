def rotate_string(text, rotation):
    if rotation > len(text):
        return text[::-1]
    split_point = len(text) - rotation
    return text[split_point:] + text[:split_point]

s = str(x)
return rotate_string(s, shift)
