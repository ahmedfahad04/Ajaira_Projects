class Hero:
    def __init__(self, heroName, hp, attack, defense, currentLevel=1):
        self.heroName = heroName
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.currentLevel = currentLevel
        self.experience = 0

    def inflictDamage(self, target):
        damage = max(self.attack - target.defense, 1)
        target.hp -= damage

    def heal(self):
        self.hp += 10
        if self.hp > 100:
            self.hp = 100
        return self.hp

    def gainExperience(self, amount):
        while amount != 0:
            if self.experience + amount >= self.currentLevel * 100:
                amount -= (self.currentLevel * 100 - self.experience)
                self.levelUp()
            else:
                self.experience += amount
                amount = 0

    def levelUp(self):
        if self.currentLevel < 100:
            self.currentLevel += 1
            self.experience = 0
            self.hp += 20
            self.attack += 5
            self.defense += 5
        return self.currentLevel, self.hp, self.attack, self.defense

    def isAlive(self):
        return self.hp > 0
