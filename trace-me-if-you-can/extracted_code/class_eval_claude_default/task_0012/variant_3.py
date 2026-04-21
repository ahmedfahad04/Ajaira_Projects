import random

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self._value = self._calculate_base_value()
        self.is_ace = rank == 'A'
    
    def _calculate_base_value(self):
        if self.rank.isdigit():
            return int(self.rank)
        elif self.rank in ['J', 'Q', 'K']:
            return 10
        else:  # Ace
            return 11
    
    def __str__(self):
        return self.rank + self.suit

class HandEvaluator:
    @staticmethod
    def evaluate(cards):
        base_value = sum(card._value for card in cards)
        ace_count = sum(1 for card in cards if card.is_ace)
        
        optimized_value = base_value
        while optimized_value > 21 and ace_count > 0:
            optimized_value -= 10
            ace_count -= 1
        
        return optimized_value

class BlackjackGame:
    def __init__(self):
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.evaluator = HandEvaluator()

    def create_deck(self):
        suits = ['S', 'C', 'D', 'H']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        deck = []
        for suit in suits:
            for rank in ranks:
                card = Card(rank, suit)
                deck.append(str(card))
        
        random.shuffle(deck)
        return deck

    def calculate_hand_value(self, hand):
        card_objects = [Card(card[:-1], card[-1]) for card in hand]
        return self.evaluator.evaluate(card_objects)

    def check_winner(self, player_hand, dealer_hand):
        player_score = self.calculate_hand_value(player_hand)
        dealer_score = self.calculate_hand_value(dealer_hand)
        
        game_state = {
            'player_bust': player_score > 21,
            'dealer_bust': dealer_score > 21,
            'player_higher': player_score > dealer_score
        }
        
        if game_state['player_bust'] and game_state['dealer_bust']:
            return 'Player wins' if player_score <= dealer_score else 'Dealer wins'
        elif game_state['player_bust']:
            return 'Dealer wins'
        elif game_state['dealer_bust']:
            return 'Player wins'
        else:
            return 'Player wins' if game_state['player_higher'] else 'Dealer wins'
