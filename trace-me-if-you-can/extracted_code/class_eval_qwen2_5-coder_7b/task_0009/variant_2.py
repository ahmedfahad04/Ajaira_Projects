class LargeNumberArithmetic:
    @staticmethod
    def add_numbers(num1, num2):
        max_len = max(len(num1), len(num2))
        num1 = num1.zfill(max_len)
        num2 = num2.zfill(max_len)

        carry = 0
        result = []
        for i in range(max_len - 1, -1, -1):
            digit_sum = int(num1[i]) + int(num2[i]) + carry
            carry = digit_sum // 10
            result.insert(0, str(digit_sum % 10))

        if carry:
            result.insert(0, str(carry))

        return ''.join(result)

    @staticmethod
    def subtract_numbers(num1, num2):
        if len(num1) < len(num2):
            num1, num2 = num2, num1
            is_negative = True
        elif len(num1) > len(num2):
            is_negative = False
        else:
            if num1 < num2:
                num1, num2 = num2, num1
                is_negative = True
            else:
                is_negative = False

        max_len = max(len(num1), len(num2))
        num1 = num1.zfill(max_len)
        num2 = num2.zfill(max_len)

        borrow = 0
        result = []
        for i in range(max_len - 1, -1, -1):
            digit_diff = int(num1[i]) - int(num2[i]) - borrow

            if digit_diff < 0:
                digit_diff += 10
                borrow = 1
            else:
                borrow = 0

            result.insert(0, str(digit_diff))

        while len(result) > 1 and result[0] == '0':
            result.pop(0)

        if is_negative:
            result.insert(0, '-')

        return ''.join(result)

    @staticmethod
    def multiply_numbers(num1, num2):
        len1, len2 = len(num1), len(num2)
        result = [0] * (len1 + len2)

        for i in range(len1 - 1, -1, -1):
            for j in range(len2 - 1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                p1, p2 = i + j, i + j + 1
                total = mul + result[p2]

                result[p1] += total // 10
                result[p2] = total % 10

        start = 0
        while start < len(result) - 1 and result[start] == 0:
            start += 1

        return ''.join(map(str, result[start:]))
