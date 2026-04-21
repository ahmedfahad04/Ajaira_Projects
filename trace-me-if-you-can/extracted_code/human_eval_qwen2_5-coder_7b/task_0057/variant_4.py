def verify_sorted_order(arr):
    return arr == sorted(arr) or arr == sorted(arr, reverse=True)
