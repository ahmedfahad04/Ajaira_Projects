import random

class BlackjackGame:
    CARD_VALUES = {
        'A': (11, True), '2': (2, False), '3': (3, False), '4': (4, False),
        '5': (5, False), '6': (6, False), '7': (7, False), '8': (8, False),
        '9': (9, False), '10': (10, False), 'J': (10, False), 
        'Q': (10, False), 'K': (10, False)
    }
    
    def __init__(self):
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        return self._build_and_shuffle_deck()

    def _build_and_shuffle_deck(self):
        suits = ['S', 'C', 'D', 'H']
        ranks = list(self.CARD_VALUES.keys())
        deck = [rank + suit for suit in suits for rank in ranks]
        random.shuffle(deck)
        return deck

    def calculate_hand_value(self, hand):
        if not hand:
            return 0
        
        total, aces = self._sum_card_values(hand)
        return self._handle_ace_adjustment(total, aces)

    def _sum_card_values(self, hand):
        total = 0
        ace_count = 0
        
        for card in hand:
            rank = card[:-1]
            value, is_ace = self.CARD_VALUES[rank]
            total += value
            if is_ace:
                ace_count += 1
                
        return total, ace_count

    def _handle_ace_adjustment(self, total, ace_count):
        while total > 21 and ace_count > 0:
            total -= 10
            ace_count -= 1
        return total

    def check_winner(self, player_hand, dealer_hand):
        player_val = self.calculate_hand_value(player_hand)
        dealer_val = self.calculate_hand_value(dealer_hand)
        
        return self._determine_winner_by_scores(player_val, dealer_val)

    def _determine_winner_by_scores(self, player_score, dealer_score):
        scenarios = {
            (True, True): lambda p, d: 'Player wins' if p <= d else 'Dealer wins',
            (True, False): lambda p, d: 'Dealer wins',
            (False, True): lambda p, d: 'Player wins',
            (False, False): lambda p, d: 'Dealer wins' if p <= d else 'Player wins'
        }
        
        player_bust = player_score > 21
        dealer_bust = dealer_score > 21
        scenario_key = (player_bust, dealer_bust)
        
        return scenarios[scenario_key](player_score, dealer_score)
