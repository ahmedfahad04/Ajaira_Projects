def solution(lst):
    def is_prime(n, divisor=2):
        if n < 2:
            return False
        if divisor > int(n**0.5):
            return True
        if n % divisor == 0:
            return False
        return is_prime(n, divisor + 1)
    
    def find_max_prime(arr, index=0, current_max=0):
        if index >= len(arr):
            return current_max
        
        if arr[index] > current_max and is_prime(arr[index]):
            current_max = arr[index]
        
        return find_max_prime(arr, index + 1, current_max)
    
    max_prime = find_max_prime(lst)
    return sum(int(d) for d in str(max_prime))
