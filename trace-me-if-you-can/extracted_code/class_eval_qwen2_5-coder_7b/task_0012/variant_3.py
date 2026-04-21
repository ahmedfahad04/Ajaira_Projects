import random

   class BlackjackGame:
       def __init__(self):
           self.card_stack = self.assemble_deck()
           self.player_deck = []
           self.dealer_deck = []

       def assemble_deck(self):
           deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4
           random.shuffle(deck)
           return deck

       def assess_hand_total(self, hand):
           total = 0
           ace_tracker = hand.count('A')
           for card in hand:
               if card.isdigit():
                   total += int(card)
               elif card in ['J', 'Q', 'K']:
                   total += 10
               else:
                   total += 11
           while total > 21 and ace_tracker > 0:
               total -= 10
               ace_tracker -= 1
           return total

       def decide_winner(self, player_deck, dealer_deck):
           player_total = self.assess_hand_total(player_deck)
           dealer_total = self.assess_hand_total(dealer_deck)
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
