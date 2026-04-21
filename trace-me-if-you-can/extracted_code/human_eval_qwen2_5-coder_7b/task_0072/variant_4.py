def check_symmetry(sequence):
       if sum(sequence) > w:
           return False

       for start_index in range(len(sequence) // 2):
           end_index = len(sequence) - start_index - 1
           if sequence[start_index] != sequence[end_index]:
               return False

       return True
