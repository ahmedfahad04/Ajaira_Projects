if not arr:
       return []
   min_even = float('inf')
   min_index = -1
   for i, x in enumerate(arr):
       if x % 2 == 0 and x < min_even:
           min_even = x
           min_index = i
   if min_index == -1:
       return []
   return [min_even, min_index]
