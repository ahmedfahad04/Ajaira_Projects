from datetime import datetime
import numpy as np

class MovieBookingSystem:
    def __init__(self):
        self.movies = {}

    def add_movie(self, name, price, start_time, end_time, n):
        self.movies[name] = {
            'price': price,
            'start_time': datetime.strptime(start_time, '%H:%M'),
            'end_time': datetime.strptime(end_time, '%H:%M'),
            'seats': np.zeros((n, n))
        }

    def book_ticket(self, name, seats_to_book):
        if name not in self.movies:
            return "Movie not found."
        
        movie = self.movies[name]
        # Check all seats first
        for row, col in seats_to_book:
            if movie['seats'][row][col] != 0:
                return "Booking failed."
        
        # Book all seats if available
        for row, col in seats_to_book:
            movie['seats'][row][col] = 1
        return "Booking success."

    def available_movies(self, start_time, end_time):
        start_dt = datetime.strptime(start_time, '%H:%M')
        end_dt = datetime.strptime(end_time, '%H:%M')
        
        return [name for name, movie in self.movies.items() 
                if start_dt <= movie['start_time'] and movie['end_time'] <= end_dt]
