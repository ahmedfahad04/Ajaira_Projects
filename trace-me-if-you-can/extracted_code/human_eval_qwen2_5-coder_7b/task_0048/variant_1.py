def is_palindrome(text):
       for idx in range(len(text) // 2):
           if text[idx] != text[-(idx + 1)]:
               return False
       return True
