class Fighter:
    def __init__(self, name, hit_points, offensive_power, defensive_power, current_level=1):
        self.name = name
        self.hit_points = hit_points
        self.offensive_power = offensive_power
        self.defensive_power = defensive_power
        self.current_level = current_level
        self.experience_points = 0

    def inflict_damage_on(self, opponent):
        damage = max(self.offensive_power - opponent.defensive_power, 1)
        opponent.hit_points -= damage

    def rejuvenate(self):
        self.hit_points += 10
        if self.hit_points > 100:
            self.hit_points = 100
        return self.hit_points

    def gain_experience(self, experience):
        while experience != 0:
            if self.experience_points + experience >= self.current_level * 100:
                experience -= (self.current_level * 100 - self.experience_points)
                self.ascend()
            else:
                self.experience_points += experience
                experience = 0

    def ascend(self):
        if self.current_level < 100:
            self.current_level += 1
            self.experience_points = 0
            self.hit_points += 20
            self.offensive_power += 5
            self.defensive_power += 5
        return self.current_level, self.hit_points, self.offensive_power, self.defensive_power

    def is_fighter_alive(self):
        return self.hit_points > 0
