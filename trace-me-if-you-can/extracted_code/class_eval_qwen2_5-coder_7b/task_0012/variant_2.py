import random

   class BlackjackGame:
       def __init__(self):
           self.card_deck = self.setup_deck()
           self.user_hand = []
           self.dealer_hand = []

       def setup_deck(self):
           deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4
           random.shuffle(deck)
           return deck

       def determine_hand_score(self, hand):
           score = 0
           ace_count = hand.count('A')
           for card in hand:
               if card.isdigit():
                   score += int(card)
               elif card in ['J', 'Q', 'K']:
                   score += 10
               else:
                   score += 11
           while score > 21 and ace_count > 0:
               score -= 10
               ace_count -= 1
           return score

       def resolve_winner(self, player_hand, dealer_hand):
           player_score = self.determine_hand_score(player_hand)
           dealer_score = self.determine_hand_score(dealer_hand)
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
