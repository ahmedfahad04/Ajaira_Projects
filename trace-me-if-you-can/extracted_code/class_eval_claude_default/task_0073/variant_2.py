class RPGCharacter:
    MAX_HP = 100
    MAX_LEVEL = 100
    EXP_PER_LEVEL = 100
    HEAL_AMOUNT = 10
    LEVEL_UP_STATS = {'hp': 20, 'attack_power': 5, 'defense': 5}

    def __init__(self, name, hp, attack_power, defense, level=1):
        self.name = name
        self.hp = hp
        self.attack_power = attack_power
        self.defense = defense
        self.level = level
        self.exp = 0

    def attack(self, other_character):
        damage = max(self.attack_power - other_character.defense, 1)
        other_character.hp -= damage

    def heal(self):
        self.hp += self.HEAL_AMOUNT
        if self.hp > self.MAX_HP:
            self.hp = self.MAX_HP
        return self.hp

    def gain_exp(self, amount):
        remaining_exp = amount
        while remaining_exp > 0:
            exp_needed = self.level * self.EXP_PER_LEVEL - self.exp
            if remaining_exp >= exp_needed:
                remaining_exp -= exp_needed
                self.level_up()
            else:
                self.exp += remaining_exp
                remaining_exp = 0

    def level_up(self):
        if self.level < self.MAX_LEVEL:
            self.level += 1
            self.exp = 0
            for stat, increase in self.LEVEL_UP_STATS.items():
                setattr(self, stat, getattr(self, stat) + increase)
        return self.level, self.hp, self.attack_power, self.defense

    def is_alive(self):
        return self.hp > 0
