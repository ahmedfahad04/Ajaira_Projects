class RPGCharacter:
    def __init__(self, name, hp, attack_power, defense, level=1):
        self.name = name
        self.hp = hp
        self.attack_power = attack_power
        self.defense = defense
        self.level = level
        self.exp = 0

    def attack(self, other_character):
        damage = max(self.attack_power - other_character.defense, 1)
        other_character.take_damage(damage)

    def take_damage(self, damage):
        self.hp -= damage

    def heal(self):
        original_hp = self.hp
        self.hp += 10
        self._cap_hp_at_max()
        return self.hp

    def _cap_hp_at_max(self):
        if self.hp > 100:
            self.hp = 100

    def gain_exp(self, amount):
        self.exp += amount
        self._process_level_ups()

    def _process_level_ups(self):
        while self._should_level_up():
            self._consume_exp_for_level()
            self.level_up()

    def _should_level_up(self):
        return self.exp >= self.level * 100 and self.level < 100

    def _consume_exp_for_level(self):
        self.exp -= self.level * 100

    def level_up(self):
        if self.level < 100:
            self.level += 1
            self.hp += 20
            self.attack_power += 5
            self.defense += 5
        return self.level, self.hp, self.attack_power, self.defense

    def is_alive(self):
        return self.hp > 0
