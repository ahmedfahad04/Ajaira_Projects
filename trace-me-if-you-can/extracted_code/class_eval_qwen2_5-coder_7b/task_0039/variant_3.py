import re
from collections import deque
from decimal import Decimal

class ExpressionInterpreter:
    def __init__(self):
        self.postfix_queue = deque()
        self.operator_priority = [0, 3, 2, 1, -1, 1, 0, 2]

    def interpret(self, expression):
        self.prepare(self.transform(expression))
        self.postfix_queue.reverse()

        result_queue = deque()
        while self.postfix_queue:
            token = self.postfix_queue.pop()
            if not self.is_operator(token):
                result_queue.append(token.replace("~", "-"))
            else:
                second_operand = result_queue.pop()
                first_operand = result_queue.pop()
                temp_result = self._execute(first_operand, second_operand, token)
                result_queue.append(str(temp_result))

        return float(eval("*".join(result_queue)))

    def prepare(self, expression):
        operator_stack = deque([','])
        arr = list(expression)
        current_index = 0
        count = 0

        for i, token in enumerate(arr):
            if self.is_operator(token):
                if count > 0:
                    self.postfix_queue.append("".join(arr[current_index: current_index + count]))
                peek_token = operator_stack[-1]
                if token == ')':
                    while operator_stack[-1] != '(':
                        self.postfix_queue.append(str(operator_stack.pop()))
                    operator_stack.pop()
                else:
                    while token != '(' and peek_token != ',' and self.compare(token, peek_token):
                        self.postfix_queue.append(str(operator_stack.pop()))
                        peek_token = operator_stack[-1]
                    operator_stack.append(token)

                count = 0
                current_index = i + 1
            else:
                count += 1

        if count > 1 or (count == 1 and not self.is_operator(arr[current_index])):
            self.postfix_queue.append("".join(arr[current_index: current_index + count]))

        while operator_stack[-1] != ',':
            self.postfix_queue.append(str(operator_stack.pop()))

    @staticmethod
    def is_operator(c):
        return c in {'+', '-', '*', '/', '(', ')', '%'}

    def compare(self, cur_token, peek_token):
        if cur_token == '%':
            cur_token = '/'
        if peek_token == '%':
            peek_token = '/'
        return self.operator_priority[ord(peek_token) - 40] >= self.operator_priority[ord(cur_token) - 40]

    @staticmethod
    def _execute(first_operand, second_operand, current_token):
        if current_token == '+':
            return Decimal(first_operand) + Decimal(second_operand)
        elif current_token == '-':
            return Decimal(first_operand) - Decimal(second_operand)
        elif current_token == '*':
            return Decimal(first_operand) * Decimal(second_operand)
        elif current_token == '/':
            return Decimal(first_operand) / Decimal(second_operand)
        elif current_token == '%':
            return Decimal(first_operand) % Decimal(second_operand)
        else:
            raise ValueError("Unexpected operator: {}".format(current_token))

    @staticmethod
    def transform(expression):
        expression = re.sub(r"\s+", "", expression)
        expression = re.sub(r"=$", "", expression)
        arr = list(expression)

        for i, c in enumerate(arr):
            if c == '-':
                if i == 0:
                    arr[i] = '~'
                else:
                    prev_c = arr[i - 1]
                    if prev_c in {'+', '-', '*', '/', '(', 'E', 'e'}:
                        arr[i] = '~'

        if arr[0] == '~' and (len(arr) > 1 and arr[1] == '('):
            arr[0] = '-'
            return "0" + "".join(arr)
        else:
            return "".join(arr)
