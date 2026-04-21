result = []
if array:
    endpoint_sum = array[0] + array[-1]
    if endpoint_sum % 2 == 0:
        result = sorted(array, reverse=True)
    else:
        result = sorted(array, reverse=False)
return result
