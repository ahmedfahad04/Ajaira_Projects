class BigNumCalculator:
    @staticmethod
    def add(num1, num2):
        def add_helper(s1, s2, carry, idx1, idx2):
            if idx1 < 0 and idx2 < 0:
                return str(carry) if carry else ""
            
            d1 = int(s1[idx1]) if idx1 >= 0 else 0
            d2 = int(s2[idx2]) if idx2 >= 0 else 0
            
            total = d1 + d2 + carry
            digit = total % 10
            new_carry = total // 10
            
            return add_helper(s1, s2, new_carry, idx1 - 1, idx2 - 1) + str(digit)
        
        return add_helper(num1, num2, 0, len(num1) - 1, len(num2) - 1)

    @staticmethod
    def subtract(num1, num2):
        def compare_numbers(n1, n2):
            if len(n1) != len(n2):
                return len(n1) - len(n2)
            return 1 if n1 > n2 else (-1 if n1 < n2 else 0)
        
        def subtract_helper(s1, s2, borrow, idx):
            if idx < 0:
                return ""
            
            diff = int(s1[idx]) - int(s2[idx]) - borrow
            if diff < 0:
                diff += 10
                new_borrow = 1
            else:
                new_borrow = 0
                
            return subtract_helper(s1, s2, new_borrow, idx - 1) + str(diff)
        
        is_negative = compare_numbers(num1, num2) < 0
        if is_negative:
            num1, num2 = num2, num1
            
        max_len = max(len(num1), len(num2))
        num1 = num1.zfill(max_len)
        num2 = num2.zfill(max_len)
        
        result = subtract_helper(num1, num2, 0, max_len - 1).lstrip('0') or '0'
        return '-' + result if is_negative and result != '0' else result

    @staticmethod
    def multiply(num1, num2):
        def multiply_single_digit(num, digit, shift):
            if digit == 0:
                return '0'
            
            result = []
            carry = 0
            
            for i in range(len(num) - 1, -1, -1):
                product = int(num[i]) * digit + carry
                result.append(str(product % 10))
                carry = product // 10
                
            if carry:
                result.append(str(carry))
                
            return ''.join(reversed(result)) + '0' * shift
        
        if num1 == '0' or num2 == '0':
            return '0'
            
        partial_products = []
        for i, digit in enumerate(reversed(num2)):
            partial_products.append(multiply_single_digit(num1, int(digit), i))
            
        result = '0'
        for product in partial_products:
            result = BigNumCalculator.add(result, product)
            
        return result
