class ChandrasekharSieve:
    def __init__(self, n):
        self.n = n
        self.primes = self.generate_primes()

    def generate_primes(self):
        if self.n < 2:
            return []

        composite = set()
        
        for base in range(2, int(self.n**0.5) + 1):
            if base not in composite:
                composite.update(range(base * base, self.n + 1, base))
        
        return [num for num in range(2, self.n + 1) if num not in composite]

    def get_primes(self):
        return self.primes
