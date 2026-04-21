total = sum(numbers)
   count = len(numbers)
   mean = total / count
   deviation = sum([abs(x - mean) for x in numbers]) / count
   return deviation
