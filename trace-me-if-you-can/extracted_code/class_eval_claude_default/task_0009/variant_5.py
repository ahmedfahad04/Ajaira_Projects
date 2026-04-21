class BigNumCalculator:
    @staticmethod
    def add(num1, num2):
        # Iterator-based approach with generators
        def digit_generator(num, start_idx):
            for i in range(start_idx, -1, -1):
                yield int(num[i])
        
        max_length = max(len(num1), len(num2))
        num1 = num1.zfill(max_length)
        num2 = num2.zfill(max_length)
        
        gen1 = digit_generator(num1, max_length - 1)
        gen2 = digit_generator(num2, max_length - 1)
        
        carry = 0
        digits = []
        
        for d1, d2 in zip(gen1, gen2):
            total = d1 + d2 + carry
            digits.append(total % 10)
            carry = total // 10
        
        if carry:
            digits.append(carry)
        
        return ''.join(str(d) for d in reversed(digits))

    @staticmethod
    def subtract(num1, num2):
        # Functional approach with list comprehensions
        def is_smaller(a, b):
            return len(a) < len(b) or (len(a) == len(b) and a < b)
        
        negative = is_smaller(num1, num2)
        if negative:
            num1, num2 = num2, num1
        
        max_len = len(num1)
        num2 = num2.zfill(max_len)
        
        # Convert to integer lists for easier manipulation
        digits1 = [int(d) for d in num1]
        digits2 = [int(d) for d in num2]
        
        # Perform subtraction with borrowing
        for i in range(max_len - 1, -1, -1):
            if digits1[i] < digits2[i]:
                # Find next non-zero digit to borrow from
                j = i - 1
                while j >= 0 and digits1[j] == 0:
                    digits1[j] = 9
                    j -= 1
                if j >= 0:
                    digits1[j] -= 1
                digits1[i] += 10
            digits1[i] -= digits2[i]
        
        # Convert back to string and remove leading zeros
        result_digits = [str(d) for d in digits1]
        while len(result_digits) > 1 and result_digits[0] == '0':
            result_digits.pop(0)
        
        result = ''.join(result_digits)
        return '-' + result if negative else result

    @staticmethod
    def multiply(num1, num2):
        # Karatsuba-inspired divide and conquer (simplified for clarity)
        if len(num1) == 1 and len(num2) == 1:
            return str(int(num1) * int(num2))
        
        # Use traditional method but with cleaner organization
        result_array = [0] * (len(num1) + len(num2))
        
        for i, d1 in enumerate(reversed(num1)):
            for j, d2 in enumerate(reversed(num2)):
                product = int(d1) * int(d2)
                pos = i + j
                
                result_array[pos] += product
                if result_array[pos] >= 10:
                    result_array[pos + 1] += result_array[pos] // 10
                    result_array[pos] %= 10
        
        # Find first non-zero digit
        start = len(result_array) - 1
        while start > 0 and result_array[start] == 0:
            start -= 1
        
        return ''.join(str(result_array[i]) for i in range(start, -1, -1))
