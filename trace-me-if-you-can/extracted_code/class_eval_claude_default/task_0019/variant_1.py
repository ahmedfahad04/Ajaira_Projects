class ChandrasekharSieve:
    def __init__(self, n):
        self.n = n
        self.primes = self.generate_primes()

    def generate_primes(self):
        if self.n < 2:
            return []

        is_prime = [True] * (self.n + 1)
        is_prime[0] = is_prime[1] = False

        for candidate in range(2, int(self.n**0.5) + 1):
            if is_prime[candidate]:
                for multiple in range(candidate * candidate, self.n + 1, candidate):
                    is_prime[multiple] = False

        return [num for num, prime_flag in enumerate(is_prime) if prime_flag]

    def get_primes(self):
        return self.primes
