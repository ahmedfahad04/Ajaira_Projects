class BigNumCalculator:
    @staticmethod
    def add(num1, num2):
        # Reverse and convert to digit lists for easier processing
        digits1 = [int(d) for d in reversed(num1)]
        digits2 = [int(d) for d in reversed(num2)]
        
        # Pad shorter number with zeros
        max_len = max(len(digits1), len(digits2))
        digits1.extend([0] * (max_len - len(digits1)))
        digits2.extend([0] * (max_len - len(digits2)))
        
        result_digits = []
        carry = 0
        
        for d1, d2 in zip(digits1, digits2):
            total = d1 + d2 + carry
            result_digits.append(total % 10)
            carry = total // 10
            
        if carry:
            result_digits.append(carry)
            
        return ''.join(str(d) for d in reversed(result_digits))

    @staticmethod
    def subtract(num1, num2):
        # Determine sign and ensure num1 >= num2
        is_negative = (len(num1) < len(num2)) or (len(num1) == len(num2) and num1 < num2)
        if is_negative:
            num1, num2 = num2, num1
            
        # Convert to reversed digit arrays
        digits1 = [int(d) for d in reversed(num1)]
        digits2 = [int(d) for d in reversed(num2)]
        
        # Pad with zeros
        digits2.extend([0] * (len(digits1) - len(digits2)))
        
        result_digits = []
        borrow = 0
        
        for d1, d2 in zip(digits1, digits2):
            diff = d1 - d2 - borrow
            if diff < 0:
                diff += 10
                borrow = 1
            else:
                borrow = 0
            result_digits.append(diff)
            
        # Remove leading zeros
        while len(result_digits) > 1 and result_digits[-1] == 0:
            result_digits.pop()
            
        result = ''.join(str(d) for d in reversed(result_digits))
        return '-' + result if is_negative else result

    @staticmethod
    def multiply(num1, num2):
        if num1 == '0' or num2 == '0':
            return '0'
            
        # Use dictionary to store partial results by position
        position_sums = {}
        
        for i, d1 in enumerate(reversed(num1)):
            for j, d2 in enumerate(reversed(num2)):
                product = int(d1) * int(d2)
                pos = i + j
                position_sums[pos] = position_sums.get(pos, 0) + product
                
        # Handle carries
        result_digits = []
        carry = 0
        pos = 0
        
        while pos in position_sums or carry:
            total = position_sums.get(pos, 0) + carry
            result_digits.append(total % 10)
            carry = total // 10
            pos += 1
            
        return ''.join(str(d) for d in reversed(result_digits))
