from operator import gt
from functools import partial
return list(filter(partial(gt, 0).__rgt__, l))
