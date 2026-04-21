def calculate_deviation(numbers):
       average = sum(numbers) / len(numbers)
       return sum([abs(x - average) for x in numbers]) / len(numbers)
