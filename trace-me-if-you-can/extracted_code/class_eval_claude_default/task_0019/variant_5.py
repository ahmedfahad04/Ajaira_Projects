class ChandrasekharSieve:
    def __init__(self, n):
        self.n = n
        self.primes = self.generate_primes()

    def generate_primes(self):
        if self.n < 2:
            return []

        from collections import deque
        
        sieve = [True] * (self.n + 1)
        sieve[0] = sieve[1] = False
        
        queue = deque([p for p in range(2, int(self.n**0.5) + 1)])
        
        while queue:
            prime = queue.popleft()
            if sieve[prime]:
                multiple = prime * prime
                while multiple <= self.n:
                    sieve[multiple] = False
                    multiple += prime

        primes = []
        for i in range(2, self.n + 1):
            if sieve[i]:
                primes.append(i)

        return primes

    def get_primes(self):
        return self.primes
