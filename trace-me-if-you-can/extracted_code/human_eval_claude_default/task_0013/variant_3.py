# Explicit temporary variable approach
while b > 0:
    temp = a % b
    a = b
    b = temp
return a
