# Version 2: Using itertools.product with filtering
from itertools import product
coords = [(i, j) for i, j in product(range(len(lst)), repeat=2) 
          if j < len(lst[i]) and lst[i][j] == x]
return sorted(coords, key=lambda pos: (pos[0], -pos[1]))
