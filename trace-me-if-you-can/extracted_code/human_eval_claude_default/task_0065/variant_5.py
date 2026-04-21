s = str(x)
chars = list(s)
if shift > len(chars):
    chars.reverse()
    return ''.join(chars)
else:
    rotation_point = len(chars) - shift
    rotated = chars[rotation_point:] + chars[:rotation_point]
    return ''.join(rotated)
