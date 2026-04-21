sum_of_numbers = sum(numbers)
   number_count = len(numbers)
   average = sum_of_numbers / number_count
   deviance = sum([abs(x - average) for x in numbers]) / number_count
   return deviance
