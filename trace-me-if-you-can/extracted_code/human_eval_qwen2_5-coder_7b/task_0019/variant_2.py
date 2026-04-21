number_to_digit = {
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9'
}

def order_numbers(input_str):
    return ' '.join(sorted(input_str.split(' '), key=lambda num: int(number_to_digit[num])))
