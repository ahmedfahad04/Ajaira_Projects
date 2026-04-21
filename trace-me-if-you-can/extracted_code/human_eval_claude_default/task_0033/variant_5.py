l = list(l)
def sort_at_step(arr, step):
    temp_dict = {}
    for i in range(0, len(arr), step):
        temp_dict[i] = arr[i]
    sorted_values = sorted(temp_dict.values())
    for i, (key, _) in enumerate(temp_dict.items()):
        arr[key] = sorted_values[i]
    return arr
return sort_at_step(l, 3)
