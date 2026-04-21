import operator

class Calculator:
    def __init__(self):
        self.ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '^': operator.pow
        }
        self.precedence_map = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    def calculate(self, expression):
        def tokenize():
            tokens = []
            current_num = ''
            for char in expression:
                if char.isdigit() or char == '.':
                    current_num += char
                else:
                    if current_num:
                        tokens.append(float(current_num))
                        current_num = ''
                    if char in '+-*/^()':
                        tokens.append(char)
            if current_num:
                tokens.append(float(current_num))
            return tokens

        def evaluate_postfix(postfix):
            stack = []
            for token in postfix:
                if isinstance(token, (int, float)):
                    stack.append(token)
                else:
                    b, a = stack.pop(), stack.pop()
                    stack.append(self.ops[token](a, b))
            return stack[0] if stack else None

        def infix_to_postfix(tokens):
            output = []
            op_stack = []
            
            for token in tokens:
                if isinstance(token, (int, float)):
                    output.append(token)
                elif token == '(':
                    op_stack.append(token)
                elif token == ')':
                    while op_stack and op_stack[-1] != '(':
                        output.append(op_stack.pop())
                    op_stack.pop()
                elif token in self.ops:
                    while (op_stack and op_stack[-1] != '(' and 
                           self.precedence_map.get(op_stack[-1], 0) >= self.precedence_map[token]):
                        output.append(op_stack.pop())
                    op_stack.append(token)
            
            while op_stack:
                output.append(op_stack.pop())
            
            return output

        tokens = tokenize()
        postfix = infix_to_postfix(tokens)
        return evaluate_postfix(postfix)
