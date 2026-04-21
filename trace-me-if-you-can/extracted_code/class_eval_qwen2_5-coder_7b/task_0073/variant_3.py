class PlayerCharacter:
    def __init__(self, playerName, health, attack, defense, characterLevel=1):
        self.playerName = playerName
        self.health = health
        self.attack = attack
        self.defense = defense
        self.characterLevel = characterLevel
        self.experience = 0

    def damageInflicted(self, enemy):
        damage = max(self.attack - enemy.defense, 1)
        enemy.health -= damage

    def healSelf(self):
        self.health += 10
        if self.health > 100:
            self.health = 100
        return self.health

    def earnXP(self, xp):
        while xp != 0:
            if self.experience + xp >= self.characterLevel * 100:
                xp -= (self.characterLevel * 100 - self.experience)
                self.levelUp()
            else:
                self.experience += xp
                xp = 0

    def levelUp(self):
        if self.characterLevel < 100:
            self.characterLevel += 1
            self.experience = 0
            self.health += 20
            self.attack += 5
            self.defense += 5
        return self.characterLevel, self.health, self.attack, self.defense

    def isAlive(self):
        return self.health > 0
