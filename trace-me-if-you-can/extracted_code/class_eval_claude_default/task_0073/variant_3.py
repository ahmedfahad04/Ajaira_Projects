from dataclasses import dataclass
from typing import Tuple

@dataclass
class RPGCharacter:
    name: str
    hp: int
    attack_power: int
    defense: int
    level: int = 1
    exp: int = 0

    def attack(self, other_character: 'RPGCharacter') -> None:
        damage = max(self.attack_power - other_character.defense, 1)
        other_character.hp -= damage

    def heal(self) -> int:
        self.hp = min(self.hp + 10, 100)
        return self.hp

    def gain_exp(self, amount: int) -> None:
        total_exp = self.exp + amount
        
        while total_exp >= self.level * 100 and self.level < 100:
            total_exp -= self.level * 100
            self.level_up()
        
        self.exp = total_exp if self.level < 100 else 0

    def level_up(self) -> Tuple[int, int, int, int]:
        if self.level < 100:
            self.level += 1
            self.exp = 0
            self.hp += 20
            self.attack_power += 5
            self.defense += 5
        return self.level, self.hp, self.attack_power, self.defense

    def is_alive(self) -> bool:
        return self.hp > 0
