total_sum = sum(numbers)
   element_count = len(numbers)
   central_tendency = total_sum / element_count
   dispersion = sum([abs(x - central_tendency) for x in numbers]) / element_count
   return dispersion
