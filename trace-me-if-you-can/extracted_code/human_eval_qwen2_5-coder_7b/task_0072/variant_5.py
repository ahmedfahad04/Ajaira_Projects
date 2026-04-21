def verify_symmetry(sequence):
       if sum(sequence) > w:
           return False

       midpoint = len(sequence) // 2
       for position in range(midpoint):
           if sequence[position] != sequence[-(position + 1)]:
               return False

       return True
