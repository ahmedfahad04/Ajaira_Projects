import random
from functools import reduce
from operator import add

class BlackjackGame:
    def __init__(self):
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        suits = ['S', 'C', 'D', 'H']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        deck = list(map(lambda x: x[1] + x[0], 
                       [(suit, rank) for suit in suits for rank in ranks]))
        random.shuffle(deck)
        return deck

    def calculate_hand_value(self, hand):
        def get_card_value(card):
            rank = card[:-1]
            if rank.isdigit():
                return int(rank), 0
            elif rank in ['J', 'Q', 'K']:
                return 10, 0
            else:  # rank == 'A'
                return 11, 1

        values_and_aces = list(map(get_card_value, hand))
        total_value = reduce(add, [val for val, _ in values_and_aces], 0)
        ace_count = reduce(add, [aces for _, aces in values_and_aces], 0)
        
        return self._optimize_ace_values(total_value, ace_count)

    def _optimize_ace_values(self, value, aces):
        return value - (10 * min(aces, max(0, (value - 21 + 9) // 10)))

    def check_winner(self, player_hand, dealer_hand):
        scores = tuple(map(self.calculate_hand_value, [player_hand, dealer_hand]))
        player_value, dealer_value = scores
        
        bust_states = tuple(map(lambda x: x > 21, scores))
        player_bust, dealer_bust = bust_states
        
        winner_map = {
            (True, True): 'Player wins' if player_value <= dealer_value else 'Dealer wins',
            (True, False): 'Dealer wins',
            (False, True): 'Player wins',
            (False, False): 'Dealer wins' if player_value <= dealer_value else 'Player wins'
        }
        
        return winner_map[bust_states]
