class PrimeCalculator:
    def __init__(self, max_value):
        self.max_value = max_value
        self.prime_numbers = self.calculate_primes()

    def calculate_primes(self):
        if self.max_value < 2:
            return []

        primes = [True] * (self.max_value + 1)
        primes[0] = primes[1] = False

        num = 2
        while num * num <= self.max_value:
            if primes[num]:
                primes[num * num::num] = [False] * ((self.max_value - num * num) // num + 1)
            num += 1

        return [i for i, prime in enumerate(primes) if prime]

    def obtain_primes(self):
        return self.prime_numbers
