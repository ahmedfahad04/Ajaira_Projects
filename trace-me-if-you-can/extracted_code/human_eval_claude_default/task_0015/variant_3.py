def generate_sequence(limit):
    current = 0
    sequence = []
    while current <= limit:
        sequence.append(str(current))
        current += 1
    return ' '.join(sequence)

return generate_sequence(n)
