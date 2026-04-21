class ChandrasekharSieve:
    def __init__(self, n):
        self.n = n
        self.primes = self.generate_primes()

    def generate_primes(self):
        def sieve_recursive(candidates, divisor_limit):
            if not candidates or divisor_limit * divisor_limit > max(candidates):
                return candidates
            
            if divisor_limit in candidates:
                filtered = [x for x in candidates if x < divisor_limit * divisor_limit or x % divisor_limit != 0]
                return sieve_recursive(filtered, divisor_limit + 1)
            else:
                return sieve_recursive(candidates, divisor_limit + 1)

        if self.n < 2:
            return []
        
        initial_candidates = list(range(2, self.n + 1))
        return sieve_recursive(initial_candidates, 2)

    def get_primes(self):
        return self.primes
