class PrimeFinder:
    def __init__(self, limit):
        self.limit = limit
        self.sieve = self.build_sieve()

    def build_sieve(self):
        if self.limit < 2:
            return []

        sieve = [True] * (self.limit + 1)
        sieve[0] = sieve[1] = False

        current_prime = 2
        while current_prime * current_prime <= self.limit:
            if sieve[current_prime]:
                for multiple in range(current_prime * current_prime, self.limit + 1, current_prime):
                    sieve[multiple] = False
            current_prime += 1

        return [num for num, is_prime in enumerate(sieve) if is_prime]

    def retrieve_primes(self):
        return self.sieve
