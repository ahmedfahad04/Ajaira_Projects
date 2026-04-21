class BattleCharacter:
    def __init__(self, charName, health, power, defense, level=1):
        self.charName = charName
        self.health = health
        self.power = power
        self.defense = defense
        self.level = level
        self.exp = 0

    def dealDamage(self, opponent):
        damage = max(self.power - opponent.defense, 1)
        opponent.health -= damage

    def selfHeal(self):
        self.health += 10
        if self.health > 100:
            self.health = 100
        return self.health

    def gainExp(self, amount):
        while amount != 0:
            if self.exp + amount >= self.level * 100:
                amount -= (self.level * 100 - self.exp)
                self.levelUp()
            else:
                self.exp += amount
                amount = 0

    def levelUp(self):
        if self.level < 100:
            self.level += 1
            self.exp = 0
            self.health += 20
            self.power += 5
            self.defense += 5
        return self.level, self.health, self.power, self.defense

    def isAlive(self):
        return self.health > 0
