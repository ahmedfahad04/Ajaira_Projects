from collections import Counter
    count_dict = Counter(numbers)
    return list(filter(lambda x: count_dict[x] == 1, numbers))
