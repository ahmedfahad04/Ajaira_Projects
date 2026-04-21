digit_transform = {
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

def arrange_numbers(text):
    return ' '.join(sorted(text.split(' '), key=lambda n: int(digit_transform[n])))
