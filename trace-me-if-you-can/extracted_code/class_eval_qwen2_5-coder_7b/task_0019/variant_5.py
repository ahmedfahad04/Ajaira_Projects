class PrimeNumberGenerator:
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        self.prime_list = self.compute_primes()

    def compute_primes(self):
        if self.upper_bound < 2:
            return []

        is_prime = [True] * (self.upper_bound + 1)
        is_prime[0] = is_prime[1] = False

        divisor = 2
        while divisor * divisor <= self.upper_bound:
            if is_prime[divisor]:
                for multiple in range(divisor * divisor, self.upper_bound + 1, divisor):
                    is_prime[multiple] = False
            divisor += 1

        primes = [number for number, prime in enumerate(is_prime) if prime]
        return primes

    def retrieve_primes(self):
        return self.prime_list
