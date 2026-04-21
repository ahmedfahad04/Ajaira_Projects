digits = filter(str.isdigit, s.split(' '))
return n - sum(map(int, digits))
