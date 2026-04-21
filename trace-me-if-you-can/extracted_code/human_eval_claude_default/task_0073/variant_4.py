return sum(a != b for a, b in zip(arr[:len(arr)//2], arr[:len(arr)//2-1:-1]))
