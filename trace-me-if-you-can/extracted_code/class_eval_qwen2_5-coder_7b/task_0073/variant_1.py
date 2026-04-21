class GameCharacter:
    def __init__(self, character_name, health_points, strike_power, armor, character_level=1):
        self.character_name = character_name
        self.health_points = health_points
        self.strike_power = strike_power
        self.armor = armor
        self.character_level = character_level
        self.experience = 0

    def inflict_damage(self, target):
        damage = max(self.strike_power - target.armor, 1)
        target.health_points -= damage

    def self_heal(self):
        self.health_points += 10
        if self.health_points > 100:
            self.health_points = 100
        return self.health_points

    def earn_experience(self, amount):
        while amount != 0:
            if self.experience + amount >= self.character_level * 100:
                amount -= (self.character_level * 100 - self.experience)
                self.level_up()
            else:
                self.experience += amount
                amount = 0

    def level_up(self):
        if self.character_level < 100:
            self.character_level += 1
            self.experience = 0
            self.health_points += 20
            self.strike_power += 5
            self.armor += 5
        return self.character_level, self.health_points, self.strike_power, self.armor

    def is_character_alive(self):
        return self.health_points > 0
