# Version 5: Using sorted() with exception handling approach
try:
    return sorted(set(lst))[1]
except IndexError:
    return None
