def check_sort_status(sequence):
    return sequence == sorted(sequence) or sequence == sorted(sequence, reverse=True)
