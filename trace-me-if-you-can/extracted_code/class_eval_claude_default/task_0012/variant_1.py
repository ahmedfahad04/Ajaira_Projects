import random
from enum import Enum
from typing import List, Tuple

class GameResult(Enum):
    PLAYER_WINS = 'Player wins'
    DEALER_WINS = 'Dealer wins'

class BlackjackGame:
    SUITS = ['S', 'C', 'D', 'H']
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    FACE_CARD_VALUE = 10
    ACE_HIGH_VALUE = 11
    ACE_LOW_VALUE = 1
    BLACKJACK_LIMIT = 21

    def __init__(self):
        self.deck = self._generate_shuffled_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        return self._generate_shuffled_deck()

    def _generate_shuffled_deck(self) -> List[str]:
        deck = [rank + suit for suit in self.SUITS for rank in self.RANKS]
        random.shuffle(deck)
        return deck

    def calculate_hand_value(self, hand: List[str]) -> int:
        total_value, ace_count = self._sum_base_values(hand)
        return self._adjust_for_aces(total_value, ace_count)

    def _sum_base_values(self, hand: List[str]) -> Tuple[int, int]:
        value = 0
        aces = 0
        for card in hand:
            rank = card[:-1]
            if rank.isdigit():
                value += int(rank)
            elif rank in ['J', 'Q', 'K']:
                value += self.FACE_CARD_VALUE
            elif rank == 'A':
                value += self.ACE_HIGH_VALUE
                aces += 1
        return value, aces

    def _adjust_for_aces(self, value: int, ace_count: int) -> int:
        while value > self.BLACKJACK_LIMIT and ace_count > 0:
            value -= (self.ACE_HIGH_VALUE - self.ACE_LOW_VALUE)
            ace_count -= 1
        return value

    def check_winner(self, player_hand: List[str], dealer_hand: List[str]) -> str:
        player_score = self.calculate_hand_value(player_hand)
        dealer_score = self.calculate_hand_value(dealer_hand)
        
        if self._both_busted(player_score, dealer_score):
            return GameResult.PLAYER_WINS.value if player_score <= dealer_score else GameResult.DEALER_WINS.value
        elif player_score > self.BLACKJACK_LIMIT:
            return GameResult.DEALER_WINS.value
        elif dealer_score > self.BLACKJACK_LIMIT:
            return GameResult.PLAYER_WINS.value
        else:
            return GameResult.DEALER_WINS.value if player_score <= dealer_score else GameResult.PLAYER_WINS.value

    def _both_busted(self, player_score: int, dealer_score: int) -> bool:
        return player_score > self.BLACKJACK_LIMIT and dealer_score > self.BLACKJACK_LIMIT
