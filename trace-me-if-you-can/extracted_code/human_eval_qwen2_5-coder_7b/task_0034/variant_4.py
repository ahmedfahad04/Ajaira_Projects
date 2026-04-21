seen = set()
   return sorted([x for x in l if not (x in seen or seen.add(x))])
