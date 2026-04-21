def compute_difference(s, n):
    number_list = list(map(int, filter(str.isdigit, s.split())))
    return n - sum(number_list)
