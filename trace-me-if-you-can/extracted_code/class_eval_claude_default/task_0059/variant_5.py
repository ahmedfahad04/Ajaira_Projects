from datetime import datetime
import numpy as np
from collections import defaultdict

class MovieBookingSystem:
    def __init__(self):
        self.movies = []
        self.seat_matrix = defaultdict(lambda: None)

    def add_movie(self, name, price, start_time, end_time, n):
        movie_record = {
            'name': name,
            'price': price,
            'start_time': datetime.strptime(start_time, '%H:%M'),
            'end_time': datetime.strptime(end_time, '%H:%M'),
            'seats': np.zeros((n, n))
        }
        self.movies.append(movie_record)
        self.seat_matrix[name] = movie_record['seats']

    def book_ticket(self, name, seats_to_book):
        seats = self.seat_matrix.get(name)
        if seats is None:
            return "Movie not found."
        
        # Check availability using exception handling for early exit
        try:
            for row, col in seats_to_book:
                if seats[row][col] != 0:
                    raise ValueError("Seat occupied")
        except ValueError:
            return "Booking failed."
        
        # Proceed with booking
        for row, col in seats_to_book:
            seats[row][col] = 1
        return "Booking success."

    def available_movies(self, start_time, end_time):
        query_start = datetime.strptime(start_time, '%H:%M')
        query_end = datetime.strptime(end_time, '%H:%M')
        
        matches = []
        for movie in self.movies:
            movie_fits = (query_start <= movie['start_time'] <= query_end and 
                         query_start <= movie['end_time'] <= query_end)
            if movie_fits:
                matches.append(movie['name'])
        return matches
