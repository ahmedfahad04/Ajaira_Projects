import random

   class BlackjackGame:
       def __init__(self):
           self.deck = self.initialize_deck()
           self.player_cards = []
           self.dealer_cards = []

       def initialize_deck(self):
           cards = []
           symbols = ['Spades', 'Clubs', 'Diamonds', 'Hearts']
           numbers = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
           for symbol in symbols:
               for number in numbers:
                   cards.append(number + ' of ' + symbol)
           random.shuffle(cards)
           return cards

       def evaluate_hand(self, hand):
           total = 0
           ace_count = 0
           for card in hand:
               card_value = self.get_card_value(card)
               total += card_value
               if card_value == 11:
                   ace_count += 1
           while total > 21 and ace_count > 0:
               total -= 10
               ace_count -= 1
           return total

       def determine_outcome(self, player_cards, dealer_cards):
           player_total = self.evaluate_hand(player_cards)
           dealer_total = self.evaluate_hand(dealer_cards)
           if player_total > 21 and dealer_total > 21:
               if player_total <= dealer_total:
                   return 'Player wins'
               else:
                   return 'Dealer wins'
           elif player_total > 21:
               return 'Dealer wins'
           elif dealer_total > 21:
               return 'Player wins'
           else:
               if player_total <= dealer_total:
                   return 'Dealer wins'
               else:
                   return 'Player wins'

       def get_card_value(self, card):
           rank = card.split()[0]
           if rank.isdigit():
               return int(rank)
           elif rank in ['Jack', 'Queen', 'King']:
               return 10
           else:
               return 11
