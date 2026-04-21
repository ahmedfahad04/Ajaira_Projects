def sort_numbers(numbers):
    class NumberWord:
        def __init__(self, word):
            self.word = word
            self.value = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'].index(word)
        
        def __lt__(self, other):
            return self.value < other.value
    
    words = [word for word in numbers.split(' ') if word]
    number_objects = [NumberWord(word) for word in words]
    number_objects.sort()
    return ' '.join(obj.word for obj in number_objects)
