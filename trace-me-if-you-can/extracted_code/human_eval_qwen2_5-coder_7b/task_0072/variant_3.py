def is_symmetric(sequence):
       if sum((item for item in sequence)) > w:
           return False

       left, right = 0, len(sequence) - 1
       while left < right:
           if sequence[left] != sequence[right]:
               return False
           left += 1
           right -= 1

       return True
