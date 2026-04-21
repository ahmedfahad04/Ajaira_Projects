import random

   class BlackjackGame:
       def __init__(self):
           self.deck_of_cards = self.make_deck()
           self.player_cards = []
           self.dealer_cards = []

       def make_deck(self):
           deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4
           random.shuffle(deck)
           return deck

       def figure_hand_value(self, hand):
           total_value = 0
           ace_counter = hand.count('A')
           for card in hand:
               if card.isdigit():
                   total_value += int(card)
               elif card in ['J', 'Q', 'K']:
                   total_value += 10
               else:
                   total_value += 11
           while total_value > 21 and ace_counter > 0:
               total_value -= 10
               ace_counter -= 1
           return total_value

       def settle_winner(self, player_cards, dealer_cards):
           player_score = self.figure_hand_value(player_cards)
           dealer_score = self.figure_hand_value(dealer_cards)
           if player_score > 21 and dealer_score > 21:
               if player_score <= dealer_score:
                   return 'Player wins'
               else:
                   return 'Dealer wins'
           elif player_score > 21:
               return 'Dealer wins'
           elif dealer_score > 21:
               return 'Player wins'
           else:
               if player_score <= dealer_score:
                   return 'Dealer wins'
               else:
                   return 'Player wins'
