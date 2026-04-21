class BigNumCalculator:
    @staticmethod
    def add(num1, num2):
        # Functional approach using generators and itertools
        from itertools import zip_longest
        
        def digit_generator(num_str):
            return (int(d) for d in reversed(num_str))
        
        gen1 = digit_generator(num1)
        gen2 = digit_generator(num2)
        
        result_stack = []
        carry = 0
        
        for d1, d2 in zip_longest(gen1, gen2, fillvalue=0):
            digit_sum = d1 + d2 + carry
            result_stack.append(str(digit_sum % 10))
            carry = digit_sum // 10
            
        if carry:
            result_stack.append(str(carry))
            
        return ''.join(reversed(result_stack))

    @staticmethod
    def subtract(num1, num2):
        # State machine approach
        class SubtractionState:
            def __init__(self, minuend, subtrahend):
                self.is_negative = self._should_negate(minuend, subtrahend)
                if self.is_negative:
                    minuend, subtrahend = subtrahend, minuend
                    
                self.minuend = minuend.zfill(max(len(minuend), len(subtrahend)))
                self.subtrahend = subtrahend.zfill(len(self.minuend))
                self.borrow = 0
                self.result_digits = []
                
            def _should_negate(self, n1, n2):
                return len(n1) < len(n2) or (len(n1) == len(n2) and n1 < n2)
                
            def process_position(self, pos):
                diff = int(self.minuend[pos]) - int(self.subtrahend[pos]) - self.borrow
                if diff < 0:
                    diff += 10
                    self.borrow = 1
                else:
                    self.borrow = 0
                self.result_digits.append(str(diff))
                
            def get_result(self):
                result = ''.join(self.result_digits).lstrip('0') or '0'
                return '-' + result if self.is_negative and result != '0' else result
        
        state = SubtractionState(num1, num2)
        for i in range(len(state.minuend) - 1, -1, -1):
            state.process_position(i)
            
        return state.get_result()

    @staticmethod
    def multiply(num1, num2):
        # Matrix-based multiplication using convolution concept
        if num1 == '0' or num2 == '0':
            return '0'
            
        # Create coefficient arrays (reversed for easier indexing)
        coeffs1 = [int(d) for d in reversed(num1)]
        coeffs2 = [int(d) for d in reversed(num2)]
        
        # Convolution: result[i+j] += coeffs1[i] * coeffs2[j]
        result_coeffs = [0] * (len(coeffs1) + len(coeffs2))
        
        for i in range(len(coeffs1)):
            for j in range(len(coeffs2)):
                result_coeffs[i + j] += coeffs1[i] * coeffs2[j]
                
        # Propagate carries
        for i in range(len(result_coeffs) - 1):
            if result_coeffs[i] >= 10:
                carry = result_coeffs[i] // 10
                result_coeffs[i] %= 10
                result_coeffs[i + 1] += carry
                
        # Remove leading zeros and convert back
        while len(result_coeffs) > 1 and result_coeffs[-1] == 0:
            result_coeffs.pop()
            
        return ''.join(str(d) for d in reversed(result_coeffs))
