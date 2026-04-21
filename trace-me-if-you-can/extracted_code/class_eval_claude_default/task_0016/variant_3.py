from collections import deque
from dataclasses import dataclass
from typing import Union

@dataclass
class Token:
    value: Union[float, str]
    type: str  # 'number', 'operator', 'lparen', 'rparen'

class Calculator:
    def __init__(self):
        self.operator_funcs = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y
        }
        self.precedence_levels = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    def calculate(self, expression):
        tokens = self._tokenize(expression)
        return self._evaluate_tokens(tokens)

    def _tokenize(self, expression):
        tokens = deque()
        i = 0
        while i < len(expression):
            char = expression[i]
            if char.isdigit() or char == '.':
                num_str = ''
                while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                    num_str += expression[i]
                    i += 1
                tokens.append(Token(float(num_str), 'number'))
            elif char in '+-*/^':
                tokens.append(Token(char, 'operator'))
                i += 1
            elif char == '(':
                tokens.append(Token(char, 'lparen'))
                i += 1
            elif char == ')':
                tokens.append(Token(char, 'rparen'))
                i += 1
            else:
                i += 1
        return tokens

    def _evaluate_tokens(self, tokens):
        values = deque()
        ops = deque()

        while tokens:
            token = tokens.popleft()
            
            if token.type == 'number':
                values.append(token.value)
            elif token.type == 'lparen':
                ops.append(token.value)
            elif token.type == 'rparen':
                while ops and ops[-1] != '(':
                    self._process_operation(values, ops)
                ops.pop()  # Remove '('
            elif token.type == 'operator':
                while (ops and ops[-1] != '(' and 
                       self.precedence_levels.get(ops[-1], 0) >= self.precedence_levels[token.value]):
                    self._process_operation(values, ops)
                ops.append(token.value)

        while ops:
            self._process_operation(values, ops)

        return values[0] if values else None

    def _process_operation(self, values, ops):
        op = ops.pop()
        b = values.pop()
        a = values.pop()
        result = self.operator_funcs[op](a, b)
        values.append(result)

    def precedence(self, operator):
        return self.precedence_levels.get(operator, 0)

    def apply_operator(self, operand_stack, operator_stack):
        operator = operator_stack.pop()
        operand2 = operand_stack.pop()
        operand1 = operand_stack.pop()
        result = self.operator_funcs[operator](operand1, operand2)
        operand_stack.append(result)
        return operand_stack, operator_stack
