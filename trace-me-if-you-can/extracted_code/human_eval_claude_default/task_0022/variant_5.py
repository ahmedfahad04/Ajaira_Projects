integers = []
non_integers = []
for x in values:
    (integers if isinstance(x, int) else non_integers).append(x)
return integers
