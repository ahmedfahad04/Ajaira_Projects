# Version 5: While loop with decrementing counter
candidate = n - 1
while candidate > 0:
    if n % candidate == 0:
        return candidate
    candidate -= 1
return 1
