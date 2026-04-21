aggregate_sum = sum(numbers)
   sample_size = len(numbers)
   arithmetic_mean = aggregate_sum / sample_size
   variance = sum([abs(x - arithmetic_mean) for x in numbers]) / sample_size
   return variance
