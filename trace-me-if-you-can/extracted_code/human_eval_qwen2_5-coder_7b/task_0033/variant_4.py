def partition_sort(input_list):
    input_list = list(input_list)
    partitioned = []
    for i in range(0, len(input_list), 3):
        partitioned.append(sorted(input_list[i:i+3]))
    flat_list = [item for sublist in partitioned for item in sublist]
    return flat_list
