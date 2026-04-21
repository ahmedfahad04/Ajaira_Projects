def join_range(start, end):
    if start == end:
        return str(end)
    else:
        return str(start) + ' ' + join_range(start + 1, end)

return join_range(0, n)
