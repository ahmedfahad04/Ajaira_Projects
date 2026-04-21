class PrimeSequence:
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        self.primes = self.find_primes()

    def find_primes(self):
        if self.upper_bound < 2:
            return []

        sieve = [True] * (self.upper_bound + 1)
        sieve[0] = sieve[1] = False

        divisor = 2
        while divisor * divisor <= self.upper_bound:
            if sieve[divisor]:
                for multiple in range(divisor * divisor, self.upper_bound + 1, divisor):
                    sieve[multiple] = False
            divisor += 1

        return [index for index, status in enumerate(sieve) if status]

    def list_primes(self):
        return self.primes
