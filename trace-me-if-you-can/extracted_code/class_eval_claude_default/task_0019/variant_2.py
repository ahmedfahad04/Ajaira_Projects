class ChandrasekharSieve:
    def __init__(self, n):
        self.n = n
        self.primes = self.generate_primes()

    def generate_primes(self):
        if self.n < 2:
            return []

        sieve = [True] * (self.n + 1)
        sieve[0] = sieve[1] = False

        def mark_multiples(prime):
            start = prime * prime
            while start <= self.n:
                sieve[start] = False
                start += prime

        current = 2
        while current * current <= self.n:
            if sieve[current]:
                mark_multiples(current)
            current += 1

        result = []
        for index in range(len(sieve)):
            if sieve[index]:
                result.append(index)
        
        return result

    def get_primes(self):
        return self.primes
