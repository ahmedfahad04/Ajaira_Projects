unique = set()
   result = []
   for item in l:
       if item not in unique:
           unique.add(item)
           result.append(item)
   return result
