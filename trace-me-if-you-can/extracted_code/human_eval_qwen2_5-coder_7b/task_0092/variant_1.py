def is_valid_triangle(a, b, c):
       if all(isinstance(i, int) for i in [a, b, c]):
           if a + b == c or a + c == b or b + c == a:
               return True
       return False
