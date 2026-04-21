class BigNumCalculator:
    @staticmethod
    def add(num1, num2):
        # Reverse and pad approach
        num1, num2 = num1[::-1], num2[::-1]
        max_len = max(len(num1), len(num2))
        num1 += '0' * (max_len - len(num1))
        num2 += '0' * (max_len - len(num2))
        
        carry = 0
        result = []
        for i in range(max_len):
            total = int(num1[i]) + int(num2[i]) + carry
            result.append(str(total % 10))
            carry = total // 10
        
        if carry:
            result.append(str(carry))
        
        return ''.join(result[::-1])

    @staticmethod
    def subtract(num1, num2):
        # Compare magnitude first approach
        def compare_magnitude(a, b):
            if len(a) != len(b):
                return len(a) - len(b)
            return 1 if a > b else -1 if a < b else 0
        
        negative = compare_magnitude(num1, num2) < 0
        if negative:
            num1, num2 = num2, num1
        
        # Reverse for easier processing
        num1, num2 = num1[::-1], num2[::-1]
        num2 += '0' * (len(num1) - len(num2))
        
        borrow = 0
        result = []
        for i in range(len(num1)):
            diff = int(num1[i]) - int(num2[i] if i < len(num2) else '0') - borrow
            if diff < 0:
                diff += 10
                borrow = 1
            else:
                borrow = 0
            result.append(str(diff))
        
        # Remove leading zeros
        while len(result) > 1 and result[-1] == '0':
            result.pop()
        
        result = ''.join(result[::-1])
        return '-' + result if negative else result

    @staticmethod
    def multiply(num1, num2):
        # Grade school multiplication with string accumulation
        if num1 == '0' or num2 == '0':
            return '0'
        
        result = '0'
        for i, digit2 in enumerate(reversed(num2)):
            partial = '0'
            carry = 0
            temp_result = []
            
            for digit1 in reversed(num1):
                product = int(digit1) * int(digit2) + carry
                temp_result.append(str(product % 10))
                carry = product // 10
            
            if carry:
                temp_result.append(str(carry))
            
            partial = ''.join(temp_result[::-1]) + '0' * i
            result = BigNumCalculator.add(result, partial)
        
        return result
