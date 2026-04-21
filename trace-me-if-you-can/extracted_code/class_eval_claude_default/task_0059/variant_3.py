from datetime import datetime
import numpy as np
from functools import reduce

class MovieBookingSystem:
    def __init__(self):
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, n):
        movie_data = (name, price, 
                     datetime.strptime(start_time, '%H:%M'),
                     datetime.strptime(end_time, '%H:%M'),
                     np.zeros((n, n)))
        self.movies.append(movie_data)

    def book_ticket(self, name, seats_to_book):
        movie_index = next((i for i, movie in enumerate(self.movies) 
                          if movie[0] == name), -1)
        
        if movie_index == -1:
            return "Movie not found."
        
        seats = self.movies[movie_index][4]
        
        # Validate all seats are available using functional approach
        seats_available = reduce(lambda acc, seat: acc and seats[seat[0]][seat[1]] == 0, 
                               seats_to_book, True)
        
        if not seats_available:
            return "Booking failed."
        
        # Book seats using side effects
        list(map(lambda seat: seats.__setitem__((seat[0], seat[1]), 1), seats_to_book))
        return "Booking success."

    def available_movies(self, start_time, end_time):
        start_dt = datetime.strptime(start_time, '%H:%M')
        end_dt = datetime.strptime(end_time, '%H:%M')
        
        filter_func = lambda movie: start_dt <= movie[2] and movie[3] <= end_dt
        return list(map(lambda movie: movie[0], 
                       filter(filter_func, self.movies)))
