a = abs(a)
from decimal import Decimal, getcontext
getcontext().prec = 50
cube_root = int(float(Decimal(a) ** (Decimal(1)/Decimal(3))).quantize(Decimal('1')))
return cube_root ** 3 == a
