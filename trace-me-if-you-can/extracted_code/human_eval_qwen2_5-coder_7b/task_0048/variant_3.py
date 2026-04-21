def check_palindrome(text):
       return all(text[i] == text[-(i + 1)] for i in range(len(text) // 2))
