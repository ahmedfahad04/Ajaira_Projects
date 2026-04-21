def sort_numbers(numbers):
    if not numbers.strip():
        return ''
    
    number_words = []
    for word in numbers.split(' '):
        if word:
            number_words.append(word)
    
    def get_numeric_value(word):
        values = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
                 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
        return values[word]
    
    number_words.sort(key=get_numeric_value)
    return ' '.join(number_words)
