result = {}
   if not arr:
       result['min_even'] = []
       result['min_index'] = []
   else:
       evens = [x for x in arr if x % 2 == 0]
       if not evens:
           result['min_even'] = []
           result['min_index'] = []
       else:
           min_even = min(evens)
           min_index = arr.index(min_even)
           result['min_even'] = [min_even]
           result['min_index'] = [min_index]
   return result['min_even'] + result['min_index']
