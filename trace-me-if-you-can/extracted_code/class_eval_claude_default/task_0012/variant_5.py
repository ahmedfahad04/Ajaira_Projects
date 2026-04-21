import random
from collections import namedtuple

GameState = namedtuple('GameState', ['player_value', 'dealer_value', 'player_bust', 'dealer_bust'])

class BlackjackGame:
    def __init__(self):
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        deck_generator = self._deck_card_generator()
        shuffled_deck = list(deck_generator)
        random.shuffle(shuffled_deck)
        return shuffled_deck

    def _deck_card_generator(self):
        suits = ['S', 'C', 'D', 'H']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        for suit in suits:
            for rank in ranks:
                yield rank + suit

    def calculate_hand_value(self, hand):
        hand_analysis = self._analyze_hand_composition(hand)
        return self._compute_optimal_value(hand_analysis)

    def _analyze_hand_composition(self, hand):
        composition = {'value': 0, 'aces': 0}
        
        for card in hand:
            rank = card[:-1]
            card_contribution = self._get_card_contribution(rank)
            composition['value'] += card_contribution['points']
            composition['aces'] += card_contribution['ace_count']
            
        return composition

    def _get_card_contribution(self, rank):
        if rank.isdigit():
            return {'points': int(rank), 'ace_count': 0}
        elif rank in ['J', 'Q', 'K']:
            return {'points': 10, 'ace_count': 0}
        else:  # rank == 'A'
            return {'points': 11, 'ace_count': 1}

    def _compute_optimal_value(self, composition):
        current_value = composition['value']
        available_aces = composition['aces']
        
        while current_value > 21 and available_aces > 0:
            current_value -= 10
            available_aces -= 1
            
        return current_value

    def check_winner(self, player_hand, dealer_hand):
        game_state = self._evaluate_game_state(player_hand, dealer_hand)
        return self._resolve_winner(game_state)

    def _evaluate_game_state(self, player_hand, dealer_hand):
        player_val = self.calculate_hand_value(player_hand)
        dealer_val = self.calculate_hand_value(dealer_hand)
        
        return GameState(
            player_value=player_val,
            dealer_value=dealer_val,
            player_bust=player_val > 21,
            dealer_bust=dealer_val > 21
        )

    def _resolve_winner(self, state):
        if state.player_bust and state.dealer_bust:
            return 'Player wins' if state.player_value <= state.dealer_value else 'Dealer wins'
        elif state.player_bust:
            return 'Dealer wins'
        elif state.dealer_bust:
            return 'Player wins'
        else:
            return 'Dealer wins' if state.player_value <= state.dealer_value else 'Player wins'
