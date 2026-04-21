try:
    next(e for e in l if e >= t)
    return False
except StopIteration:
    return True
