def validate_palindrome(text):
       midpoint = len(text) // 2
       for i in range(midpoint):
           if text[i] != text[-(i + 1)]:
               return False
       return True
