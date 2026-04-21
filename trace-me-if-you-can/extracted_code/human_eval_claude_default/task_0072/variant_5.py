# Variant 5: Stack-based validation approach
if sum(q) > w:
    return False

stack = []
mid = len(q) // 2

# Push first half onto stack
for i in range(mid):
    stack.append(q[i])

# Compare with second half
start_idx = mid + (len(q) % 2)  # Skip middle element if odd length
for i in range(start_idx, len(q)):
    if not stack or stack.pop() != q[i]:
        return False

return len(stack) == 0
