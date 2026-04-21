seen = set(l)
for i in range(len(l)):
    for j in range(i + 1, len(l)):
        target = -(l[i] + l[j])
        if target in seen and target != l[i] and target != l[j]:
            # Additional check to ensure we're not reusing the same index
            remaining = l[j+1:]
            if target in remaining:
                return True
return False
