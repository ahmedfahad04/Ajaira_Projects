import random

   class BlackjackGame:
       def __init__(self):
           self.playing_cards = self.build_deck()
           self.player_cards = []
           self.dealer_cards = []

       def build_deck(self):
           deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4
           random.shuffle(deck)
           return deck

       def evaluate_card_value(self, card):
           if card.isdigit():
               return int(card)
           elif card in ['J', 'Q', 'K']:
               return 10
           else:
               return 11

       def calculate_hand_value(self, hand):
           total_value = 0
           ace_count = hand.count('A')
           for card in hand:
               total_value += self.evaluate_card_value(card)
           while total_value > 21 and ace_count > 0:
               total_value -= 10
               ace_count -= 1
           return total_value

       def determine_hand_outcome(self, player_hand, dealer_hand):
           player_value = self.calculate_hand_value(player_hand)
           dealer_value = self.calculate_hand_value(dealer_hand)
           if player_value > 21 and dealer_value > 21:
               if player_value <= dealer_value:
                   return 'Player wins'
               else:
                   return 'Dealer wins'
           elif player_value > 21:
               return 'Dealer wins'
           elif dealer_value > 21:
               return 'Player wins'
           else:
               if player_value <= dealer_value:
                   return 'Dealer wins'
               else:
                   return 'Player wins'
