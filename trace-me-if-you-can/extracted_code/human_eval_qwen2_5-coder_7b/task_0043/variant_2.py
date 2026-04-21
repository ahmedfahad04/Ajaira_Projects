def check_for_zero_sums(sequence):
    for pos, value in enumerate(sequence):
        for next_pos in range(pos + 1, len(sequence)):
            if value + sequence[next_pos] == 0:
                return True
    return False
