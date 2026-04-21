def is_arranged(sequence):
    return sequence == sorted(sequence) or sequence == sorted(sequence, reverse=True)
