class RPGCharacter:
    def __init__(self, name, hp, attack_power, defense, level=1):
        self._stats = {
            'name': name,
            'hp': hp,
            'attack_power': attack_power,
            'defense': defense,
            'level': level,
            'exp': 0
        }

    def __getattr__(self, name):
        if name in self._stats:
            return self._stats[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == '_stats':
            super().__setattr__(name, value)
        else:
            self._stats[name] = value

    def attack(self, other_character):
        damage_dealt = self.attack_power - other_character.defense
        other_character.hp -= 1 if damage_dealt <= 0 else damage_dealt

    def heal(self):
        healed_hp = self.hp + 10
        self.hp = healed_hp if healed_hp <= 100 else 100
        return self.hp

    def gain_exp(self, amount):
        exp_to_process = amount
        while exp_to_process and self.level < 100:
            exp_needed_for_next_level = self.level * 100 - self.exp
            if exp_to_process >= exp_needed_for_next_level:
                exp_to_process -= exp_needed_for_next_level
                self.level_up()
            else:
                self.exp += exp_to_process
                exp_to_process = 0

    def level_up(self):
        if self.level < 100:
            self.level += 1
            self.exp = 0
            self.hp += 20
            self.attack_power += 5
            self.defense += 5
        return self.level, self.hp, self.attack_power, self.defense

    def is_alive(self):
        return self.hp > 0
