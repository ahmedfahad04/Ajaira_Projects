def is_symmetric(sequence):
       if sum(sequence) > w:
           return False

       for index in range(len(sequence) // 2):
           if sequence[index] != sequence[-(index + 1)]:
               return False

       return True
