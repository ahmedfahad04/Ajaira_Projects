# Variant 4: While loop with mathematical approach
def count_sevens_v4(n):
    result = 0
    i = 0
    
    while i < n:
        if i % 11 == 0 or i % 13 == 0:
            # Count 7s in current number
            temp_num = i
            while temp_num > 0:
                if temp_num % 10 == 7:
                    result += 1
                temp_num //= 10
            # Handle the case when i is 0
            if i == 0:
                pass  # 0 has no 7s
        i += 1
    
    return result
