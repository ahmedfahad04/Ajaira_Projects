result = []
   for str in strings:
       if str[:len(prefix)] == prefix:
           result.append(str)
   return result
