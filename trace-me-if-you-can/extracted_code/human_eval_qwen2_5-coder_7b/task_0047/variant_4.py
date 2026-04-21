def calculate_median(array):
    array.sort()
    mid_index = len(array) // 2
    if len(array) % 2:
        return array[mid_index]
    else:
        return (array[mid_index - 1] + array[mid_index]) / 2.0
