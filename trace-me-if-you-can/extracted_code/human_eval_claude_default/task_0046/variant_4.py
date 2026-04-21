# Version 4: Circular buffer with modular arithmetic
base_values = {0: 0, 1: 0, 2: 2, 3: 0}
if n in base_values:
    return base_values[n]

buffer = [0, 0, 2, 0]
current_pos = 0

for i in range(4, n + 1):
    new_value = sum(buffer)
    buffer[current_pos] = new_value
    current_pos = (current_pos + 1) % 4

return buffer[(current_pos - 1) % 4]
