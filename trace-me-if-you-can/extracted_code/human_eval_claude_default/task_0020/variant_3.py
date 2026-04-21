# Variant 3: Divide and conquer approach with sorted array
def find_closest_pair(numbers):
    if len(numbers) < 2:
        return None
    
    def closest_pair_rec(sorted_nums):
        n = len(sorted_nums)
        if n <= 3:
            min_dist = float('inf')
            best_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    dist = abs(sorted_nums[i] - sorted_nums[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = tuple(sorted([sorted_nums[i], sorted_nums[j]]))
            return best_pair, min_dist
        
        mid = n // 2
        left_pair, left_dist = closest_pair_rec(sorted_nums[:mid])
        right_pair, right_dist = closest_pair_rec(sorted_nums[mid:])
        
        if left_dist <= right_dist:
            return left_pair, left_dist
        else:
            return right_pair, right_dist
    
    sorted_numbers = sorted(numbers)
    result, _ = closest_pair_rec(sorted_numbers)
    return result
