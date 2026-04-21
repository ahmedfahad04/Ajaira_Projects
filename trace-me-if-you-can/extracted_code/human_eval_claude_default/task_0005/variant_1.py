if not numbers:
    return []

return [item for i, num in enumerate(numbers) 
        for item in ([num, delimeter] if i < len(numbers) - 1 else [num])]
