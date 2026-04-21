if not strings:
    return None

return next((s for s in strings if len(s) == max(len(x) for x in strings)), None)
