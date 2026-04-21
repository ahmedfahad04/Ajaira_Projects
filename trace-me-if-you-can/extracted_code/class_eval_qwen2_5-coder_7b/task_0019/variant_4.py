class SieveOfEratosthenes:
    def __init__(self, max_number):
        self.max_number = max_number
        self.prime_list = self.get_primes()

    def get_primes(self):
        if self.max_number < 2:
            return []

        prime_flags = [True] * (self.max_number + 1)
        prime_flags[0] = prime_flags[1] = False

        current_number = 2
        while current_number * current_number <= self.max_number:
            if prime_flags[current_number]:
                prime_flags[current_number * current_number::current_number] = [False] * ((self.max_number - current_number * current_number) // current_number + 1)
            current_number += 1

        return [num for num, flag in enumerate(prime_flags) if flag]

    def fetch_primes(self):
        return self.prime_list
